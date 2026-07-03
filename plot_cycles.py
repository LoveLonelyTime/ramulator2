#!/usr/bin/env python3
"""
按 cmd_list.sh 里的方法各跑一遍：先生成 mem.txt，再跑 example_config.py 得到 result.txt。
从 result.txt 里读：
  - cycles      : 32 个 controller 的最大 cycles（整仿真结束时刻）
  - total_BW    : 32 个 controller 的 read_throughput_MBps 之和 → GB/s
以最慢方法为 baseline，其它方法 speedup = baseline_cycles / method_cycles。
柱高 = speedup，柱上方标注 speedup / cycles / bandwidth。
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MEM_TXT = ROOT / "mem.txt"
RESULT_TXT = ROOT / "result.txt"
SIM_SCRIPT = ROOT / "examples" / "example_config.py"

COMMON = ["--pe-num", "32", "--tile-num", "128",
          "--batch", "4", "--token-num", "2048",
          "--bits", "4", "-o", str(MEM_TXT)]

METHODS = [
    ("KIVI",       ["gen_mem_trace_kivi.py"]                          + COMMON),
    ("KIVI 128g",  ["gen_mem_trace_kivi.py",   "--group-size", "128"] + COMMON),
    ("KVQuant",    ["gen_mem_trace_kvquant.py"]                       + COMMON),
    ("Atom",       ["gen_mem_trace_atom.py"]                          + COMMON),
    ("Qserve",     ["gen_mem_trace_qserve.py"]                        + COMMON),
    ("ADKV",       ["gen_mem_trace_adkv.py"]                          + COMMON),
    ("SKVQ",       ["gen_mem_trace_skvq.py"]                          + COMMON),
    ("AxCore",     ["gen_mem_trace_kivi.py",   "--group-size", "64"]  + COMMON),
    ("Tender",     ["gen_mem_trace_qserve.py"]                        + COMMON),
]


def run(cmd, stdout=None):
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT, stdout=stdout, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"cmd failed: {' '.join(cmd)}")


def sim_stats():
    """跑 ramulator，返回 (cycles, total_bw_GBps)。"""
    with open(RESULT_TXT, "w") as f:
        run([sys.executable, str(SIM_SCRIPT)], stdout=f)
    with open(RESULT_TXT) as f:
        stats = json.load(f)
    ctrls = stats["memory_system"]["controller"]
    cycles = max(c["cycles"] for c in ctrls)
    bw_MBps = sum(c["read_throughput_MBps"] for c in ctrls)
    return cycles, bw_MBps / 1000.0


def main():
    labels, cycles_list, bw_list = [], [], []
    for label, argv in METHODS:
        run([sys.executable] + argv)
        c, bw = sim_stats()
        print(f"  -> {label}: {c:,} cycles, {bw:.1f} GB/s")
        labels.append(label)
        cycles_list.append(c)
        bw_list.append(bw)

    baseline = max(cycles_list)
    speedups = [baseline / c for c in cycles_list]

    order = sorted(range(len(labels)), key=lambda i: speedups[i])
    labels     = [labels[i]     for i in order]
    speedups   = [speedups[i]   for i in order]
    cycles_list = [cycles_list[i] for i in order]
    bw_list    = [bw_list[i]    for i in order]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(labels, speedups, color="#4C72B0", edgecolor="black")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", label="baseline (slowest)")
    ax.set_ylabel("Speedup  (baseline / this)")
    ax.set_title("KV-cache quantization: speedup vs. slowest\n"
                 f"(baseline = {baseline:,} cycles)")
    ax.set_ylim(0, max(speedups) * 1.25)
    for bar, s, c, bw in zip(bars, speedups, cycles_list, bw_list):
        ax.text(bar.get_x() + bar.get_width() / 2,
                s + max(speedups) * 0.01,
                f"{s:.2f}×\n{c:,} cyc\n{bw:.1f} GB/s",
                ha="center", va="bottom", fontsize=8.5)
    ax.legend(loc="upper left")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    out = ROOT / "speedup_bar.png"
    plt.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
