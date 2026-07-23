#!/usr/bin/env python3
"""
Compare SA-ADKV (systolic-array accelerator running the ADKV trace) against
GPU-ADKV throughput at low batch sizes.

Metric
------
throughput_reqs_per_s = B * T / seconds, with T = 2048, NH = 32, DIM = 128.

SA-ADKV
-------
- generate ADKV memory trace at each B (pe/tile fixed)
- feed into examples/example_config.py with MEM=gddr6, bits=4, burst=64
- cycles → seconds = cycles / FREQ_HZ → throughput = B*T / seconds

GPU
---
- read gpu_bench.csv row where backend == "ADKV (4-bit)", T == 2048, B ∈ BATCHES

Cache: fig_batch_gpu_cache.json.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


import matplotlib.font_manager as fm

font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

# 手动添加字体到管理器
fm.fontManager.addfont(font_path)

# 从文件创建 FontProperties 获取准确名称
prop = fm.FontProperties(fname=font_path)
font_name = prop.get_name()

# 设置全局字体
plt.rcParams['font.family'] = font_name

def compute(groups, batch = 64, head = 32, dim = 128, context_length = 4096, metadata_per_group = 4, bytes_per_ele = 0.5):
    quantized_sizes = []
    metadata_sizes = []
    for group in groups:
        quantized_size = 2 * batch * head * dim * context_length * bytes_per_ele
        metadata_size = 2 * batch * head * dim * context_length / group * metadata_per_group
        total = quantized_size + metadata_size
        quantized_sizes.append(quantized_size / total * 100)
        metadata_sizes.append(metadata_size / total * 100)
    return quantized_sizes, metadata_sizes



ROOT = Path(__file__).resolve().parent
MEM_TXT = ROOT / "mem.txt"
CACHE_FILE = ROOT / "fig_batch_gpu_cache.json"
GPU_CSV = ROOT / "gpu_bench.csv"

# ---- workload ----
BATCHES = [1, 4, 16, 32]
TOKEN_NUM = 2048
HEAD = 32
DIM = 128
BITS = 4

# ---- SA-ADKV hardware ----
# controller.cycles counts memory-clock ticks; convert with tCK_ps.
MEM = "gddr6"
BURST_SIZE = 64          # GDDR6/DDR class
TCK_PS = 570             # GDDR6_2000_1350mV_double → 1.754 GHz mem
TILE_NUM = 128
PE_NUM = 32

# Theoretical peak DRAM bandwidth for utilization %.
# SA-ADKV side (GDDR6, derived from example_config.py):
#   32 controllers × (tx_bytes / (nBL × tCK_ps))
#   = 32 × (64 B / (4 × 570 ps)) ≈ 898.25 GB/s
SA_PEAK_BW_MBPS = 898_246
# GPU side (target device datasheet, e.g. A100/H100 HBM budget).
GPU_PEAK_BW_MBPS = 864_000

ADKV_SCRIPT = "gen_mem_trace_adkv.py"


# ---- helpers ------------------------------------------------------------

def run_gen(batch: int) -> None:
    argv = [
        sys.executable, str(ROOT / ADKV_SCRIPT),
        "--pe-num", str(PE_NUM),
        "--tile-num", str(TILE_NUM),
        "--batch", str(batch),
        "--head", str(HEAD),
        "--dim", str(DIM),
        "--token-num", str(TOKEN_NUM),
        "--bits", str(BITS),
        "--burst-size", str(BURST_SIZE),
        "-o", str(MEM_TXT),
    ]
    r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"cmd failed: {' '.join(argv)}")


def run_sim() -> tuple:
    """Return (max cycles, total measured throughput in MB/s)."""
    env = os.environ.copy()
    env["MEM"] = MEM
    r = subprocess.run(
        [sys.executable, "examples/example_config.py"],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit("simulator failed")
    stats = json.loads(r.stdout)
    ctrls = stats["memory_system"]["controller"]
    cycles = max(int(c["cycles"]) for c in ctrls)
    total_mbps = sum(float(c.get("total_throughput_MBps", 0.0)) for c in ctrls)
    return cycles, total_mbps


def sa_key(batch: int) -> str:
    return f"sa|{MEM}|b{BITS}|T{TOKEN_NUM}|B{batch}|pe{PE_NUM}|tile{TILE_NUM}"


def collect_sa(force: bool) -> dict:
    cache = {}
    if CACHE_FILE.exists() and not force:
        cache = json.loads(CACHE_FILE.read_text())
    for B in BATCHES:
        key = sa_key(B)
        if key in cache:
            continue
        print(f"[gen] SA-ADKV mem={MEM} bits={BITS} B={B}"
              f" pe={PE_NUM} tile={TILE_NUM}")
        run_gen(B)
        cycles, bw_mbps = run_sim()
        # controller.cycles is in memory-clock ticks (tCK_ps period).
        seconds = cycles * TCK_PS * 1e-12
        throughput = B * TOKEN_NUM / seconds
        cache[key] = {
            "batch": B, "cycles": cycles,
            "throughput": throughput,
            "bw_mbps": bw_mbps,
            "bw_util": bw_mbps / SA_PEAK_BW_MBPS,
        }
        CACHE_FILE.write_text(json.dumps(cache, indent=2))
    return cache


def load_gpu() -> dict:
    """Return {B: {"throughput": ..., "bw_util": ...}} for ADKV (4-bit),
    T=2048. Throughput = B*T / (latency_ms * 1e-3); BW = ncu_bytes_total /
    (latency_ms * 1e-3), utilization = BW / GPU_PEAK_BW."""
    out = {}
    with open(GPU_CSV) as f:
        for row in csv.DictReader(f):
            if row["backend"] != "ADKV (4-bit)":
                continue
            if int(row["T"]) != TOKEN_NUM:
                continue
            if int(row["NH"]) != HEAD or int(row["DIM"]) != DIM:
                continue
            B = int(row["B"])
            if B not in BATCHES:
                continue
            latency_s = float(row["latency_ms"]) * 1e-3
            throughput = B * TOKEN_NUM / latency_s
            bytes_total = float(row["ncu_bytes_total"])
            bw_mbps = bytes_total / latency_s / 1e6   # B/s → MB/s
            out[B] = {
                "throughput": throughput,
                "bw_mbps": bw_mbps,
                "bw_util": bw_mbps / GPU_PEAK_BW_MBPS,
            }
    missing = [B for B in BATCHES if B not in out]
    if missing:
        raise SystemExit(f"gpu_bench.csv missing B={missing} at T={TOKEN_NUM}")
    return out


# ---- plotting -----------------------------------------------------------

def plot(sa_cache: dict, gpu: dict, outpath: Path):
    sa = {sa_cache[sa_key(B)]["batch"]: sa_cache[sa_key(B)]["throughput"]
          for B in BATCHES}
    sa_util = {B: sa_cache[sa_key(B)].get("bw_util", 0.0) for B in BATCHES}

    x = np.arange(len(BATCHES))
    width = 0.25
    norm = gpu[1]["throughput"]
    sa_vals = [sa[B] / norm for B in BATCHES]
    gpu_vals = [gpu[B]["throughput"] / norm for B in BATCHES]
    sa_util_vals = [sa_util[B] * 100 for B in BATCHES]
    gpu_util_vals = [gpu[B]["bw_util"] * 100 for B in BATCHES]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax2 = ax.twinx()

    b1 = ax.bar(x - width / 2, sa_vals, width,
                color="#4E79A7",
                label="ADKV-SA", zorder=2)
    b2 = ax.bar(x + width / 2, gpu_vals, width,
                color="#CEE1EF", 
                label="ADKV-GPU", zorder=2)

    # DRAM bandwidth utilization on right Y (percent), one line each.
    l1, = ax2.plot(x, sa_util_vals, linewidth=4,
                   color="#e61c5d", zorder=3, alpha=0.9,
                   label="ADKV-SA BW Util.")
    l2, = ax2.plot(x, gpu_util_vals, linewidth=4,
                   color="#F088A9", zorder=3,  alpha=0.9,
                   label="ADKV-GPU BW Util.")
    
    ax2.scatter(
        x,
        sa_util_vals,
        color="#e61c5d",
        s=80,
        marker="o",
        facecolors='white',
        linewidths=2,
        label=f'',
        zorder=10
    )

    ax2.scatter(
        x,
        gpu_util_vals,
        color="#F088A9",
        s=80,
        marker="o",
        facecolors='white',
        linewidths=2,
        label=f'',
        zorder=10,
    )

    for bars, vals in [(b1, sa_vals), (b2, gpu_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.2f}",
                    ha="center", va="bottom", fontsize=11, zorder=10)

    for xi, v in zip(x, sa_util_vals):
        ax2.text(xi - 0.05, v - 10.5, f"{v:.0f}%",
                 ha="right", va="bottom", fontsize=11, color="#e61c5d")
    for xi, v in zip(x, gpu_util_vals):
        ax2.text(xi - 0.05, v + 3.5, f"{v:.0f}%",
                 ha="left", va="bottom", fontsize=11, color="#F088A9")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{B}" for B in BATCHES], fontsize=11)
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylim(0, 25)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("Normalized Throughput (Tokens/s)", fontsize=11)
    ax2.set_ylabel(
        f"DRAM Bandwidth Util.",
        fontsize=11,
    )
    ax2.set_ylim(0, 100)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4, zorder=0)

    handles = [b1, b2, l1, l2]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1), frameon=False, fontsize=11, ncol=4)

    ax2.tick_params(axis="y", labelsize=11)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _p: f"{v:.0f}%"))

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    print(f"saved {outpath}")


# ---- main ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="ignore cache and rerun SA simulation")
    args = ap.parse_args()

    sa_cache = collect_sa(force=args.force)
    gpu = load_gpu()
    plot(sa_cache, gpu, ROOT / "fig_batch_gpu.pdf")


if __name__ == "__main__":
    main()
