#!/usr/bin/env python3
"""
ADKV QK DDR/HBM Memory Access Trace Generator

Generates memory access traces (LD <address>) for the ADKV QK hardware accelerator,
suitable for feeding into DDR/HBM memory simulators.

Memory Layout (DRAM):
  Q_BASE    --- Q[B][H][C]          FP16 (2 bytes each)
  PARAM_BASE -- Params[H][C][5]     16-bit (2 bytes each): dc_init, scale_init, alpha, beta, t
  K_BASE    --- K[B][H][C][N]       INT4 (0.5 bytes each), Channel-major

Usage:
  # Single trace generation
  python3 gen_mem_trace.py trace --pe-num 128 --dim 1024 --token-num 2048 -o trace.txt

  # Sweep configurations and output CSV statistics
  python3 gen_mem_trace.py sweep --output-csv sweep_results.csv

  # Stats-only mode (no trace file, just print statistics)
  python3 gen_mem_trace.py trace --pe-num 128 --dim 1024 --token-num 2048 --stats-only
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
    param_idx: 0=dc_init, 1=scale_init, 2=alpha, 3=beta, 4=t
    """
    offset = (h * args.dim * 5 + c * 5 + param_idx) * 2
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

def addr_zp(args: Namespace, b: int, h: int, c: int, t: int, group_size: int = 32) -> int:
    """
    SKVQ [B, H, G, T]
    """
    group_num = args.dim // group_size
    group_id = c // group_size
    channel_base = (b * args.head * group_num + h * group_num + group_id) * args.token_num *  2 # FP8
    byte_offset = t * 2
    return args.param_base + channel_base + byte_offset

def burst_align(addr: int, burst_size: int) -> int:
    """Align address down to burst boundary."""
    return addr & ~(burst_size - 1)


# =============================================================================
# Trace Generator
# =============================================================================
cache_global = set()
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

    for token_start in range(0, args.token_num, args.pe_num):
        token_end = min(token_start + args.pe_num, args.token_num)
        for c in range(channel_start, channel_end):
            # SKVQ
            start_addr = addr_zp(args, b, h, c, token_start)
            end_addr = addr_zp(args, b, h, c, token_end - 1)

            # Generate bursts covering [start_addr, end_addr]
            first_burst = burst_align(start_addr, bs)
            last_burst = burst_align(end_addr, bs)
            shared_group_size = 32
            shared_group_inner_id = tile % 32
            tile_rr = (token_start // args.pe_num) % shared_group_size
            addr = first_burst
            while addr <= last_burst:
                if tile_rr == shared_group_inner_id and addr not in cache:
                    cache.add(addr)
                    yield (tile, addr, 1, True, f"ZP b={b} h={h} c={c} tile={tile}")
                addr += bs

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
