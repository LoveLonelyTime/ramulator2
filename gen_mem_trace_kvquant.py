#!/usr/bin/env python3
"""
DDR/HBM Memory Access Trace Generator

Generates memory access traces (LD <address>) for the hardware accelerator,
suitable for feeding into DDR/HBM memory simulators.

Memory Layout (DRAM):
  Q_BASE    --- Q[B][H][C]          FP16 (2 bytes each)
  PARAM_BASE -- Params[H][C][5]     16-bit (2 bytes each): dc_init, scale_init, alpha, beta, t
  K_BASE    --- K[B][H][C][N]       INT4 (0.5 bytes each), Channel-major
"""

import argparse
from argparse import Namespace
import csv
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import product, zip_longest
from typing import Generator, List, Optional, TextIO, Tuple


def KVQuant_memory(batch, hidden_dim, seq_len, bits=4, outlier_ratio=0.01):
    """
    KVQuant (arxiv 2401.18079) memory breakdown.

    Quantization strategy:
      Keys  → per-channel quantization  (one scale per channel, across ALL tokens)
      Values→ per-token   quantization  (one scale per token,   across ALL channels)

    Metadata (verified against KVQuant/benchmarking/scripts/test_kernels_{key,value}.py):
      1. Key   scale + zero_point  : per-channel → O(hidden_dim),  NOT O(seq_len × hidden_dim)
                                     FP32 tensors, shape (hidden_dim,) each
      2. Value scale + zero_point  : per-token   → O(seq_len),     NOT O(seq_len × hidden_dim)
                                     FP32 tensors, shape (seq_len,) each
      3. Key   NUQ lookup table    : per-channel non-uniform quantization codebook
                                     shape (hidden_dim, 2^bits), FP32
                                     → O(hidden_dim × 2^bits), independent of seq_len
      4. Value NUQ lookup table    : single shared NF4 signpost table
                                     shape (2^bits,), FP32 — negligible but included
      5. Key   sparse outliers     : dense-sparse: outlier_ratio × seq_len tokens per channel
                                     stored as FP32 value + INT32 column-index per entry,
                                     plus INT32 start_rows pointer array of size O(seq_len)
      6. Value sparse outliers     : outlier_ratio × hidden_dim channels per token,
                                     stored as FP32 value + INT32 row-index per entry,
                                     plus INT32 start_cols pointer array of size O(hidden_dim)

    This is the key structural difference from KIVI:
      KIVI meta ∝ seq_len × hidden_dim / group_size   (both dims grow)
      KVQuant meta = O(hidden_dim × 2^bits) + O(seq_len)
                   + O(outlier_ratio × seq_len × hidden_dim)
    """
    # ── Quantized KV data ─────────────────────────────────────────────────────
    key_quant_bytes = batch * hidden_dim * seq_len * bits / 8
    val_quant_bytes = batch * hidden_dim * seq_len * bits / 8

    # ── Scale + zero-point metadata ───────────────────────────────────────────
    # Keys: per-channel → one (scale, zero_point) pair per (batch, channel), FP32
    key_scale_bytes = batch * hidden_dim * 2 * 4   # 2 × FP32 per channel

    # Values: per-token → one (scale, zero_point) pair per (batch, token), FP32
    val_scale_bytes = batch * seq_len * 2 * 4       # 2 × FP32 per token

    # ── Non-uniform quantization (NUQ) lookup tables ─────────────────────────
    # Keys: per-channel LUT; shape (hidden_dim, 2^bits), FP32
    #   (from: lookup_table = torch.zeros((num_heads, head_dim, 2**4)), test_kernels_key.py)
    key_lut_bytes = batch * hidden_dim * (2 ** bits) * 4   # FP32 per entry

    # Values: single shared NF4 signpost table; shape (2^bits,), FP32
    #   (from: lookup_table = torch.tensor(nf4_signposts).float(), test_kernels_value.py)
    val_lut_bytes = (2 ** bits) * 4   # negligible; one shared table regardless of batch

    # ── Sparse outlier metadata (dense-sparse quantization) ───────────────────
    # Actual code uses FP32 vals and INT32 indices (torch default dtypes):
    #   rows2/cols2 = int32 (4 B each), vals2 = float32 (4 B)
    # Key COO: (col_index INT32 + val FP32) per nnz, plus start_rows INT32 array O(seq_len)
    key_nnz = batch * hidden_dim * seq_len * outlier_ratio
    key_outlier_bytes = key_nnz * (4 + 4) + batch * seq_len * 4   # nnz*(idx+val) + start_rows

    # Value COO: (row_index INT32 + val FP32) per nnz, plus start_cols INT32 array O(hidden_dim)
    val_nnz = batch * hidden_dim * seq_len * outlier_ratio
    val_outlier_bytes = val_nnz * (4 + 4) + batch * hidden_dim * 4  # nnz*(idx+val) + start_cols

    quant_bytes = key_quant_bytes + val_quant_bytes
    meta_bytes  = (key_scale_bytes + val_scale_bytes
                   + key_lut_bytes + val_lut_bytes
                   + key_outlier_bytes + val_outlier_bytes)
    return quant_bytes, meta_bytes

# =============================================================================
# Address Computation
# =============================================================================
def addr_q(args: Namespace, b: int, h: int, c: int) -> int:
    """
    Compute byte address for Q[b][h][c].
    Layout: Q[batch][head][channel], each element FP16 (2 bytes).
    """
    offset = (b * args.head * args.dim + h * args.dim + c) * 2
    return args.q_base + offset


def addr_param(args: Namespace, h: int, c: int, param_idx: int) -> int:
    """
    Compute byte address for Params[h][c][param_idx].
    Layout: Params[head][channel][5], each element 16-bit (2 bytes).
    param_idx: 0=dc_init, 1=scale_init
    """
    offset = (h * args.dim * 18 + c * 18 + param_idx) * 4
    return args.param_base + offset


def addr_k(args: Namespace, b: int, h: int, c: int, token_offset: int) -> int:
    """
    Compute byte address for K[b][h][c][token_offset].
    Layout: K[batch][head][channel][token], Channel-major, INT4 packed.
    Two INT4 values per byte, token_offset in elements (not bytes).
    Returns byte address of the byte containing this INT4 element.
    """
    # Base for this (b, h, c) channel stripe
    atom_ratio = 128
    channel_base = (b * args.head * args.dim + h * args.dim + c) * ((args.token_num - atom_ratio) // 2)
    # Byte offset within the channel stripe
    byte_offset = token_offset // 2
    return args.k_base + channel_base + byte_offset

def addr_outlier_ptr(args: Namespace, b: int, t: int) -> int:
    """
    Compute byte address for K outlier start_rows[b][t].
    Layout: start_rows[batch][token_num+1], INT32 (4 bytes each).
    The +1 entry per batch lets us read start_rows[token_end] to bound the last range.
    Note: shared across heads (per KVQuant: meta size is O(seq_len), not O(head*seq_len)).
    """
    offset = (b * args.token_num + t) * 4
    return args.outlier_ptr_base + offset


def addr_outlier_data(args: Namespace, b: int, entry_idx: int) -> int:
    """
    Compute byte address for K outlier entry [b][entry_idx].
    Each entry: (col_idx INT32 + val FP32) = 8 bytes.
    Layout: per-batch contiguous array sized to max_nnz = ceil(token_num*head*dim*outlier_ratio).
    Entries are ordered by token (CSR by token, col_idx = channel within hidden_dim).
    """
    max_nnz_per_batch = math.ceil(args.token_num * args.head * args.dim * args.outlier_ratio)
    offset = (b * max_nnz_per_batch + entry_idx) * 8
    return args.outlier_data_base + offset


def addr_k_INT8(args: Namespace, b: int, h: int, c: int, token_offset: int) -> int:
    """
    Compute byte address for K[b][h][c][token_offset].
    Layout: K[batch][head][channel][token], Channel-major, INT4 packed.
    Two INT4 values per byte, token_offset in elements (not bytes).
    Returns byte address of the byte containing this INT4 element.
    """
    # Base for this (b, h, c) channel stripe
    atom_ratio = 128
    channel_base = (b * args.head * args.dim + h * args.dim + c) * (atom_ratio)
    # Byte offset within the channel stripe
    byte_offset = token_offset
    return 0x0600_0000 + channel_base + byte_offset

def burst_align(addr: int, burst_size: int) -> int:
    """Align address down to burst boundary."""
    return addr & ~(burst_size - 1)


# =============================================================================
# Trace Generator
# =============================================================================
param_cache = set()
def generate_tiled_trace(args: Namespace, tile: int, b: int, h: int, channel_start: int, channel_end: int) -> Generator[Tuple[int, int ,int, bool, str], None, None]:
    """
    Generate memory access trace for a single tiled PEs.
    """
    bs = args.burst_size
    cache = set()
    for c in range(channel_start, channel_end):
        # --- 1. Load Q[b][h][c] ---
        q_addr = burst_align(addr_q(args, b, h, c), bs)
        if q_addr not in cache:
            cache.add(q_addr)
            yield (tile, q_addr, 1, True, f"Q b={b} h={h} c={c}")

        # --- 2. Load ZP, SC, LUT(16) ---
        for p_idx in range(18):
            p_addr = burst_align(addr_param(args, h, c, p_idx), bs)
            if p_addr not in param_cache:
                param_cache.add(p_addr)
                yield (tile, p_addr, 1, True, f"PARAM h={h} c={c} p={p_idx}")

    for token_start in range(0, args.token_num, args.pe_num):
        token_end = min(token_start + args.pe_num, args.token_num)

        # --- 3a. Load K outlier CSR pointers (start_rows) ---
        for t in range(token_start, token_end):
            ptr_addr = burst_align(addr_outlier_ptr(args, b, t), bs)
            if ptr_addr not in param_cache:
                param_cache.add(ptr_addr)
                yield (tile, ptr_addr, 1, True,
                       f"K_OUTLIER_PTR b={b} t={t}")

        # # --- 3b. Load K outlier (col_idx, val) entries in the token range ---
        # # Sparse decoders read all entries spanning [start_rows[token_start],
        # # start_rows[token_end]) and filter by col_idx. We approximate the
        # # entry range using the expected count (outlier_ratio × hidden × tokens).
        nnz_lo = math.floor(token_start * args.head * args.dim * args.outlier_ratio)
        nnz_hi = math.ceil(token_end   * args.head * args.dim * args.outlier_ratio)
        if nnz_hi > nnz_lo:
            data_start = addr_outlier_data(args, b, nnz_lo)
            data_end   = addr_outlier_data(args, b, nnz_hi - 1)  # last byte of last entry
            first_burst = burst_align(data_start, bs)
            last_burst  = burst_align(data_end,   bs)
            addr = first_burst
            while addr <= last_burst:
                if addr not in param_cache:
                    param_cache.add(addr)
                    yield (tile, addr, 1, True,
                           f"K_OUTLIER b={b} t=[{token_start},{token_end})")
                addr += bs

        for c in range(channel_start, channel_end):
            # --- 3. Load K[b][h][c][tok_start:tok_end] ---
            # INT4 packed: active_pes elements = active_pes/2 bytes, contiguous
            k_start_addr = addr_k(args, b, h, c, token_start)
            k_end_addr = addr_k(args, b, h, c, token_end - 1)  # last element

            # Generate bursts covering [k_start_addr, k_end_addr]
            first_burst = burst_align(k_start_addr, bs)
            last_burst = burst_align(k_end_addr, bs)

            addr = first_burst
            while addr <= last_burst:
                if addr in cache:
                    yield (tile, addr, args.cache_delay + 1, False, f"K b={b} h={h} c={c} tile={tile}")
                else:
                    cache.add(addr)
                    yield (tile, addr, 1, True, f"K b={b} h={h} c={c} tile={tile}")
                addr += bs

def generate_trace(args: Namespace) -> Generator[Tuple[int, int, int, bool, str], None, None]:
    """
    Generate memory access trace in hardware execution order.

    Yields: (tile_id, burst_aligned_address, delay, issue, annotation_string)
    """
    assert args.dim % args.tile_num == 0
    channel_split = args.dim // args.tile_num
    tiled_id = 0
    # [b, h, 0:cfg.dim:channel_split] -> Tiles
    for b in range(args.batch):
        for h in range(args.head):
            for channel_start in range(0, args.dim, channel_split):
                channel_end = min(channel_start + channel_split, args.dim)
                yield from generate_tiled_trace(args, tiled_id % args.tile_num, b, h, channel_start, channel_end)
                tiled_id += 1

# =============================================================================
# Output Writers
# =============================================================================
def write_trace(args: Namespace, outfile: TextIO, annotated: bool = False) -> dict:
    """
    Write trace to file and return statistics.

    Returns: dict with burst counts per region.
    """
    counts = {}

    for id, addr, delay, issue, annotation in generate_trace(args):
        region = annotation.split()[0]
        # if region == "K_OUTLIER_PTR":
        if args.annotated:
            outfile.write(f"{id} 0x{addr:08X} {delay} {1 if issue else 0} # {annotation}\n")
        else:
            outfile.write(f"{id} 0x{addr:08X} {delay} {1 if issue else 0}\n")

        # Count by region
        if issue:
            region = annotation.split()[0]
            counts[region] = counts.get(region, 0) + 1

    total = sum(counts.values())
    return {
        "total_bursts": total,
        "total_bytes": total * args.burst_size,
        "burst_regions": counts
    }


def print_statistics(stats: dict, args: Namespace):
    """Pretty-print trace statistics."""
    print(f"\n{'='*65}")
    print(f"  Memory Trace Statistics")
    print(f"  Config: {args.pe_num}PE*{args.tile_num}TILE --- {args.batch}BATCH{args.head}HEAD{args.dim}DIM{args.token_num}TOKEN")
    print(f"{'='*65}")
    print(f"  Burst size: {args.burst_size}B")
    print(f"  Cache delay: {args.cache_delay}")
    print(f"{'─'*65}")
    print(f"  Total DRAM bursts:   {stats['total_bursts']:>12,}")
    print(f"  Total bytes read:    {stats['total_bytes']:>12,} ({stats['total_bytes']/1024/1024:.2f} MB)")
    print(f"{'─'*65}")
    total = stats['total_bursts']
    for region, bursts in stats['burst_regions'].items():
        pct = 100.0 * bursts / total if total > 0 else 0
        print(f"  {region:>6s}: {bursts:>10,} bursts, "
              f"{bursts * args.burst_size:>10,} bytes ({pct:5.1f}%)")
    print(f"{'─'*65}")

    # Theoretical time at various bandwidths
    bw_configs = [
        ("LPDDR5 (51 GB/s)", 51e9),
        ("LPDDR5x (68 GB/s)", 68e9),
        ("HBM2e (2 TB/s)", 2e12),
        ("GDDR6X8 (776 GB/s)", 776e9),
    ]
    print(f"{'─'*65}")
    print(f"  Theoretical min time (memory-bound):")
    for name, bw in bw_configs:
        time_us = stats['total_bytes'] / bw * 1e6
        print(f"    {name}: {time_us:>8.2f} us")
    print(f"{'='*65}\n")

# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="ADKV QK DDR/HBM Memory Access Trace Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--pe-num", type=int, default=32)
    parser.add_argument("--tile-num", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--token-num", type=int, default=2048)
    parser.add_argument("--cache-delay", type=int, default=2)
    parser.add_argument("--burst-size", type=int, default=64,
                         help="DRAM burst size in bytes (DDR=64, HBM=32)")
    parser.add_argument("--annotated", action="store_true",
                         help="Include region annotations in trace")
    parser.add_argument("-o", "--output", type=str, default=None,
                         help="Output trace file (default: stdout)")
    # Base addresses
    parser.add_argument("--q-base", type=lambda x: int(x, 0), default=0x0000_0000)
    parser.add_argument("--param-base", type=lambda x: int(x, 0), default=0x0100_0000)
    parser.add_argument("--k-base", type=lambda x: int(x, 0), default=0x0200_0000)
    parser.add_argument("--outlier-ptr-base", type=lambda x: int(x, 0), default=0x0700_0000,
                         help="Base address of K outlier start_rows[batch][token_num+1] INT32 array")
    parser.add_argument("--outlier-data-base", type=lambda x: int(x, 0), default=0x0800_0000,
                         help="Base address of K outlier (col_idx INT32 + val FP32) entries")
    parser.add_argument("--outlier-ratio", type=float, default=0.01,
                         help="Fraction of K entries stored as outliers (KVQuant default 1%)")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.output:
        outfile = open(args.output, "w")
    else:
        outfile = sys.stdout

    stats = write_trace(args, outfile)

    if args.output:
        outfile.close()
        # Print stats to stderr so they're visible even with file output
        print_statistics(stats, args)
    else:
        # Trace went to stdout, print stats to stderr
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        print_statistics(stats, args)
        sys.stdout = old_stdout

if __name__ == "__main__":
    main()
