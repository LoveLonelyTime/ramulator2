#!/usr/bin/env python3
"""
论文大图：2 (mem) × 4 (model) 网格 = 8 subplots
每子图：9 methods × 2 bits = 18 grouped bars
共 144 bars.

Baseline: 每子图内 4bit KIVI 的 cycles 定为 1.0，speedup = baseline / method_cycles.

Cache: 已经跑过的 (mem, model, bit, method) 结果保存到 speedup_grid_cache.json，
       支持增量补跑；删除或重命名该文件即从头跑。
"""

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



ROOT       = Path(__file__).resolve().parent
MEM_TXT    = ROOT / "mem.txt"
RESULT_TXT = ROOT / "result.txt"
SIM_SCRIPT = ROOT / "examples" / "example_config.py"
CACHE_FILE = ROOT / "speedup_grid_cache.json"

MEMORIES = ["gddr6", "hbm3"]

# 每种 mem 对应的 DRAM burst / tx 大小；trace 生成器要与之匹配
BURST_BYTES = {"gddr6": 64, "hbm3": 32}

# 组合 A：(label, batch, head, dim, token_num)
MODELS = [
    ("Llama-2 7B",       4, 32, 128,  4096),
    ("Llama-2 13B",      4, 40, 128,  4096),
    ("Llama-3 8B",       4,  8, 128,  4096),
    ("Llama-3 8B 32K",   4,  8, 128, 32768),
]

BITS = [4, 2]

# 生成器脚本以及为该方法追加的额外参数
METHODS = [
    ("KIVI",       "gen_mem_trace_kivi.py",    []),
    ("KIVI 128g",  "gen_mem_trace_kivi.py",    ["--group-size", "128"]),
    ("KVQuant",    "gen_mem_trace_kvquant.py", []),
    ("Atom",       "gen_mem_trace_atom.py",    []),
    ("Qserve",     "gen_mem_trace_qserve.py",  []),
    ("SKVQ",       "gen_mem_trace_skvq.py",    []),
    ("AxCore",     "gen_mem_trace_kivi.py",    ["--group-size", "64"]),
    ("Tender",     "gen_mem_trace_qserve.py",  []),
    ("ADKV",       "gen_mem_trace_adkv.py",    []),
]

METHOD_LABELS = [m[0] for m in METHODS]
BASELINE_METHOD = "KIVI"   # 每子图 4bit KIVI 定为 1.0


# ----------------------------------------------------------------- helpers
def run(cmd, stdout=None, env_extra=None):
    print("$", " ".join(cmd))
    env = os.environ.copy()
    # ramulator 的 python 绑定在 <root>/python，把它塞进 PYTHONPATH，
    # 也支持编译产物 <root>/build 目录
    py_extra = f"{ROOT / 'python'}:{ROOT / 'build'}"
    env["PYTHONPATH"] = py_extra + ":" + env.get("PYTHONPATH", "")
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(cmd, cwd=ROOT, stdout=stdout, stderr=subprocess.PIPE,
                       text=True, env=env)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"cmd failed: {' '.join(cmd)}")


def sim_stats(mem):
    """跑 ramulator 一次，返回 (cycles, total_bw_GBps)."""
    with open(RESULT_TXT, "w") as f:
        run([sys.executable, str(SIM_SCRIPT)],
            stdout=f, env_extra={"MEM": mem})
    with open(RESULT_TXT) as f:
        stats = json.load(f)
    ctrls = stats["memory_system"]["controller"]
    cycles = max(c["cycles"] for c in ctrls)
    bw_MBps = sum(c["read_throughput_MBps"] for c in ctrls)
    return cycles, bw_MBps / 1000.0


def gen_trace(script, batch, head, dim, tokens, bits, extra, burst_size):
    argv = [
        sys.executable, script,
        "--pe-num", "32", "--tile-num", "128",
        "--burst-size", str(burst_size),
        "--batch", str(batch), "--head", str(head),
        "--dim", str(dim), "--token-num", str(tokens),
        "--bits", str(bits),
        "-o", str(MEM_TXT),
    ] + list(extra)
    run(argv)


def load_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def cache_key(mem, model_label, bit, method):
    return f"{mem}|{model_label}|{bit}|{method}"


# ------------------------------------------------------------------- main
def main():
    cache = load_cache()

    # ---- 1. 逐个 config 跑仿真，缓存结果 ----
    for mem in MEMORIES:
        burst = BURST_BYTES[mem]
        for m in MODELS:
            model_label, batch, head, dim, tokens = m
            for bit in BITS:
                for method_label, script, extra in METHODS:
                    key = cache_key(mem, model_label, bit, method_label)
                    if key in cache:
                        continue
                    print(f"\n>> {key}")
                    gen_trace(script, batch, head, dim, tokens, bit, extra, burst)
                    cyc, bw = sim_stats(mem)
                    cache[key] = {"cycles": cyc, "bw_GBps": bw}
                    save_cache(cache)

    # ---- 2. 画图 ----
    # 两行 (mem) × 一整条 x 轴（4 model 拼在一起），model 之间只用一根竖线分隔。
    # 每 model block 内部按 4bit / 2bit 两小组排列。
    fig, axes = plt.subplots(
        len(MEMORIES), 1,
        figsize=(24, 8),
        sharex=True,
    )
    # colors = {4: "#4E79A7", 2: "#7EA6D8"}
    colors = {"KIVI": "#B0DAFF", "KIVI 128g": "#A2CCF4" , "KVQuant": "#94BEE7", "Atom": "#86B0DA", "Qserve": "#78A2CD", "SKVQ": "#6A94C0", "AxCore": "#5C86B3", "Tender": "#4E79A7", "ADKV": "#2C4F73"}
    n_methods = len(METHOD_LABELS)
    bit_gap   = 0.8                          # 4bit / 2bit 两 section 之间的间隙
    model_gap = 1.2                          # 相邻 model 之间的间隙（除去竖线）
    section_w = n_methods                    # 每个 section 占 9 位置
    model_w   = 2 * section_w + bit_gap      # 一个 model block 的宽度
    width     = 0.8

    # 计算每个 (model_idx, bit) 的 x 起点
    def bit_xs(model_idx, bit):
        base = model_idx * (model_w + model_gap)
        if bit == 4:
            return np.arange(n_methods) + base
        return np.arange(n_methods) + base + section_w + bit_gap

    def model_center(model_idx):
        base = model_idx * (model_w + model_gap)
        return base + model_w / 2 - 0.5

    def model_boundary(model_idx):
        # 相邻两 model 之间的分隔线 x 坐标：位于两 block 中点
        base = model_idx * (model_w + model_gap)
        return base + model_w + model_gap / 2 - 0.5

    all_x = []
    all_labels = []
    for mi in range(len(MODELS)):
        for bit in BITS:
            xs = bit_xs(mi, bit).tolist()
            all_x += xs
            all_labels += METHOD_LABELS

    for r, mem in enumerate(MEMORIES):
        ax = axes[r]
        y_max_row = 0
        for mi, m in enumerate(MODELS):
            model_label = m[0]
            base_key_4 = cache_key(mem, model_label, 4, BASELINE_METHOD)
            base_key_2 = cache_key(mem, model_label, 2, BASELINE_METHOD)
            baseline_cyc_4 = cache[base_key_4]["cycles"]
            baseline_cyc_2 = cache[base_key_2]["cycles"]

            speedups = {b: [] for b in BITS}
            for method_label, _, _ in METHODS:
                for bit in BITS:
                    entry = cache[cache_key(mem, model_label, bit, method_label)]
                    baseline_cyc = baseline_cyc_4 if bit == 4 else baseline_cyc_2
                    speedups[bit].append(baseline_cyc / entry["cycles"])

            y_max_row = max(y_max_row, max(speedups[4] + speedups[2]))

            for bit in BITS:
                xs = bit_xs(mi, bit)
                bars = ax.bar(xs, speedups[bit],
                              width=width, color=colors.values(),
                              alpha = 1.0 if bit == 4 else 0.7,
                              hatch = "////" if bit == 4 else "\\\\\\\\",
                              edgecolor="#1A3A6B", linewidth=0.5,
                              label=f"{bit}bit" if (mi == 0 and r == 0) else None)
                for bar, s in zip(bars, speedups[bit]):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            s + 0.015,
                            f"{s:.2f}", ha="center", va="bottom",
                            fontsize=15, rotation=90, fontweight="bold")

        # y 上限（该行统一）
        y_top = 2.0
        ax.set_ylim(0, y_top)

        # model 之间的竖分隔线
        for mi in range(len(MODELS) - 1):
            xb = model_boundary(mi)
            ax.axvline(xb, color="#666666", linewidth=1.0, linestyle="-", alpha=0.9)

        # 每个 model 顶部 title 标签
        for mi, m in enumerate(MODELS):
            ax.text(model_center(mi), -0.6, m[0],
                    ha="center", va="top",
                    fontsize=20, fontweight="bold",
                    )
            # bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#666666", lw=0.6, alpha=0.9)

        # 每个 model block 内 4bit / 2bit 小标签（低一点，避免和 model 名撞）
        if mem == "gddr6":
            for mi in range(len(MODELS)):
                base = mi * (model_w + model_gap)
                ax.text(base + section_w / 2 - 0.5, 2.2, "4-Bit",
                        ha="center", va="top", fontsize=20, fontweight="bold")
                ax.text(base + section_w + bit_gap + section_w / 2 - 0.5,
                        2.2, "2-Bit",
                        ha="center", va="top", fontsize=20, fontweight="bold")

        ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--", zorder=-1)
        ax.set_ylabel(f"Speedup - {mem.upper()}", fontsize=20)
        ax.tick_params(axis="y", labelsize=15)
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)

        # 只在最后一行画 x tick 标签
        ax.set_xticks(all_x)
        if r == len(MEMORIES) - 1:
            ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=13)
        else:
            ax.set_xticklabels([""] * len(all_x))

        ax.set_xlim(-1, (len(MODELS) - 1) * (model_w + model_gap) + model_w)

    # 全局 legend
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper center", ncol=2,
    #            fontsize=11, bbox_to_anchor=(0.5, 0.985))

    # fig.suptitle("KV-cache quantization: speedup grid  (mem × model × bit)",
    #              fontsize=13, y=1.005)
    plt.tight_layout(rect=(0, 0, 1, 0.955))
    # 减小两行之间的间距
    plt.subplots_adjust(hspace=0.06)
    out = ROOT / "speedup_grid.pdf"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
