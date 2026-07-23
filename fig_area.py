#!/usr/bin/env python3
"""
Area-efficiency curve: Tokens/s/mm^2  vs.  PE area budget (mm^2).

For each method:
  - Given an area budget A_budget (mm^2), pick (tile_num, pe_num) so that
        total_pe = tile_num * pe_num
        total_pe * area_per_pe <= A_budget
    with tile_num maximized first, then pe_num — i.e. we cap tile_num at
    TILE_MAX, then let pe_num absorb any remaining budget up to PE_MAX.
  - Regenerate mem.txt via gen_mem_trace_<method>.py using those (pe_num,
    tile_num).
  - Run examples/example_config.py to get memory_system cycles.
  - Frequency = 1 GHz (frontend), so seconds_per_step = cycles / 1e9.
    Each decode step generates BATCH output tokens, so
        tokens/s = BATCH / seconds_per_step
    and
        tokens/s/mm^2 = (BATCH / seconds_per_step) / A_used
    where A_used = total_pe * area_per_pe (actual area used, not the budget).

As DRAM saturates, tokens/s stops growing while A_used keeps growing, so
the curve rolls off.

Per-PE area (μm^2, W4/FP16) is taken from micro58-axcore
`Software/axcore_simulator/params/systolic_array_synth_W4-FP16.csv`.
The CSV headers are shifted by one column — the values labelled
"Leakage Power (nW)" are actually the per-PE Area (um^2) numbers; see the
project discussion for details.

Cache: results saved to fig_area_cache.json for fast re-runs.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
import matplotlib.colors as mcolors

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
RESULT_TXT = ROOT / "result_area.txt"
CACHE_FILE = ROOT / "fig_area_cache.json"

# ---- workload ----
BATCH = 4
HEAD = 32
DIM = 128
TOKEN_NUM = 2048
BITS_LIST = [2, 4]

# ---- hardware knobs ----
FREQ_HZ = 1e9
TILE_MAX = 128      # 优先把 tile_num 拉满，剩余预算再让 pe_num 长
PE_MAX = 1024

# ADKV Area: 3585.2712 um2
# Tender Area: 488.28 um2
# AxCore Area: 1056 um2

# Per-PE area (μm^2) at W4/FP16, 1 GHz — see file docstring.
PE_AREA_UM2 = {
    "ADKV":    2693.43,   # AxCore-style approximate PE (W4 shared-add)
    "AxCore":  1056,
    # "KIVI":    2475,   # fpma
    # "KVQuant": 2475,
    # "Qserve":  2475,
    # "Atom":    2475,
    "Tender":  488.28,
    # "SKVQ":    2475,
}
DEFAULT_PE_AREA_UM2 = 2002.53

# ---- methods ----
METHODS = [
    ("KIVI",    "gen_mem_trace_kivi.py",    []),
    ("KVQuant", "gen_mem_trace_kvquant.py", []),
    ("AxCore",  "gen_mem_trace_kivi.py",    ["--group-size", "64"]),
    ("Atom",    "gen_mem_trace_atom.py",    []),
    ("Qserve",  "gen_mem_trace_qserve.py",  []),
    # ("Tender",  "gen_mem_trace_qserve.py",    []),
    ("ADKV",    "gen_mem_trace_adkv.py",    []),
]
METHOD_LABELS = [m[0] for m in METHODS]

METHOD_COLORS = {
    "KIVI":    "#4E79A7",
    "KVQuant": "#F28E2B",
    "AxCore":  "#59A14F",
    "Atom":    "#E15759",
    "Tender": "#E15759",
    "Qserve":  "#B07AA1",
    "ADKV":    "#000000",
}

# ---- backends ----
MEMS = ["gddr6", "hbm3"]

# Area budgets (mm^2) — enough range to reach DRAM saturation.
#     0.005, 0.01, 0.02, 0.05, 0.1, 20.0, 40.0
AREA_BUDGETS_MM2 = [
    0.2, 0.5,
    1.0, 2.0, 5.0, 10.0, 20.0
]


# ---- helpers ------------------------------------------------------------

def pick_pe_tile(area_budget_mm2: float, area_per_pe_um2: float):
    """Return (tile_num, pe_num, area_used_mm2).

    Constraints:
      - tile_num must be a power of 2, capped at TILE_MAX.
      - pe_num is any positive int, capped at PE_MAX.
      - Grow tile and pe together: target tile ≈ sqrt(total) (rounded up
        to a power of 2, but no larger than TILE_MAX), then pe absorbs the
        remainder. This keeps tile_num "prioritized" (breaks ties in tile's
        favor) without letting one dimension eat the whole budget.
    """
    area_per_pe_mm2 = area_per_pe_um2 * 1e-6
    max_total = area_budget_mm2 / area_per_pe_mm2
    if max_total < 1:
        return None

    total_cap = int(max_total)
    # tile target ≈ sqrt(total_cap), rounded UP to a power of 2, then capped
    root = max(1, math.isqrt(total_cap))
    tile_target = 1 << (root - 1).bit_length() if root > 1 else 1
    tile = min(TILE_MAX, tile_target)
    while tile > 1 and tile > total_cap:
        tile //= 2
    pe = min(PE_MAX, total_cap // tile)
    pe = max(1, pe)
    pe = 315
    # pe = 50
    total = tile * pe
    return tile, pe, total * area_per_pe_mm2


def burst_size_for(mem: str) -> int:
    """DDR-class → 64 B bursts, HBM-class → 32 B bursts."""
    return 32 if mem.lower().startswith("hbm") else 64


def run_gen(script: str, pe_num: int, tile_num: int, bits: int,
            burst_size: int, extra: list) -> None:
    argv = [
        sys.executable, str(ROOT / script),
        "--pe-num", str(pe_num),
        "--tile-num", str(tile_num),
        "--batch", str(BATCH),
        "--head", str(HEAD),
        "--dim", str(DIM),
        "--token-num", str(TOKEN_NUM),
        "--bits", str(bits),
        "--burst-size", str(burst_size),
        "-o", str(MEM_TXT),
    ] + list(extra)
    r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"cmd failed: {' '.join(argv)}")


def run_sim(mem: str) -> int:
    """Run examples/example_config.py, return max controller cycles."""
    env = os.environ.copy()
    env["MEM"] = mem
    r = subprocess.run(
        [sys.executable, "examples/example_config.py"],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit("simulator failed")
    stats = json.loads(r.stdout)
    ctrls = stats["memory_system"]["controller"]
    return max(int(c["cycles"]) for c in ctrls)


def cache_key(mem: str, method: str, budget: float, bits: int) -> str:
    return f"{mem}|{method}|{budget}|b{bits}"


def collect(force: bool) -> dict:
    cache = {}
    if CACHE_FILE.exists() and not force:
        cache = json.loads(CACHE_FILE.read_text())

    for bits in BITS_LIST:
        for mem in MEMS:
            for name, script, extra in METHODS:
                area_pp = PE_AREA_UM2.get(name, DEFAULT_PE_AREA_UM2)
                for budget in [PLOT_BUDGET]:
                    key = cache_key(mem, name, budget, bits)
                    if key in cache:
                        continue
                    sized = pick_pe_tile(budget, area_pp)
                    if sized is None:
                        cache[key] = {"skipped": True}
                        CACHE_FILE.write_text(json.dumps(cache, indent=2))
                        continue
                    tile, pe, area_used = sized
                    bs = burst_size_for(mem)
                    print(f"[gen] mem={mem} method={name} bits={bits}"
                          f" budget={budget}"
                          f" -> tile={tile} pe={pe} bs={bs}"
                          f" area={area_used:.4f}mm^2")
                    run_gen(script, pe_num=pe, tile_num=tile, bits=bits,
                            burst_size=bs, extra=extra)
                    cycles = run_sim(mem)
                    cache[key] = {
                        "tile": tile, "pe": pe,
                        "area_used": area_used,
                        "cycles": cycles,
                    }
                    CACHE_FILE.write_text(json.dumps(cache, indent=2))
    return cache


def tokens_per_s(entry: dict) -> float:
    seconds_per_step = entry["cycles"] / FREQ_HZ
    return BATCH / seconds_per_step


# ---- plotting -----------------------------------------------------------
BASELINE = "KIVI"
PLOT_BUDGET = 20.0   # mm^2 — DRAM-saturated regime


def _plot_one(ax, cache, mem, bits, show_ylabel_left):
    rows = []
    for name, _, _ in METHODS:
        entry = cache.get(cache_key(mem, name, PLOT_BUDGET, bits))
        if not entry or entry.get("skipped"):
            continue
        tps = tokens_per_s(entry)
        tps_per_area = tps / entry["area_used"]
        pe_total = entry["tile"] * entry["pe"]
        rows.append((name, tps_per_area, pe_total))

    base = next((r for r in rows if r[0] == BASELINE), None)
    if base is None:
        return
    base_eff, base_pe = base[1], base[2]

    names = [r[0] for r in rows]
    eff_norm = [r[1] / base_eff for r in rows]
    pe_norm = [r[2] / base_pe for r in rows]
    # Bar colors: light-blue → dark-blue gradient by METHODS order,
    # so KIVI is lightest and ADKV is darkest.
    order = {n: i for i, n in enumerate(METHOD_LABELS)}
    n_methods = max(1, len(METHOD_LABELS) - 1)

    colors = ["#CEE1EF", "#7EA6D8", "#4E79A7"] #688DB4
    cmap = mcolors.LinearSegmentedColormap.from_list('my_gradient', colors)
    colors = [cmap((order.get(n, 0) + 1) / n_methods) for n in names]

    x = np.arange(len(names))
    bars = ax.bar(x, eff_norm, width=0.55, color=colors, zorder=2) # edgecolor="black", linewidth=0.6, 
    # ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)

    raw_max = max(max(eff_norm), 1.05)
    ymax = math.ceil(raw_max * 10) / 10 + 0.05  # small headroom for labels
    ymin = 0.9
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.round(np.arange(ymin, ymax + 1e-9, 0.1), 1))

    # Per-bar annotation: "tps×  /  PE×"
    for bar, eff, pe in zip(bars, eff_norm, pe_norm):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{eff:.2f}\n #PEs = {pe:.2f}",
                ha="center", va="bottom", fontsize=11, linespacing=1.1)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    if show_ylabel_left:
        ax.set_ylabel(f"Normalized Tokens/s/mm$^2$ - {mem.upper()}", fontsize=13)
    if mem == "gddr6":
        ax.set_title(f"{bits}-Bit", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4, zorder=0)


def plot(cache: dict, outpath: Path):
    # Rows = MEMS (GDDR6, HBM3), Cols = BITS_LIST (2, 4). 2x2 grid.
    nrows, ncols = len(MEMS), len(BITS_LIST)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(12, 8),
                             sharey=False, sharex=True)
    axes = np.atleast_2d(axes)

    for r, mem in enumerate(MEMS):
        for c, bits in enumerate(BITS_LIST):
            ax = axes[r, c]
            _plot_one(ax, cache, mem, bits, show_ylabel_left=(c == 0))

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    print(f"saved {outpath}")


# ---- main ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="ignore cache and rerun all points")
    args = ap.parse_args()

    cache = collect(force=args.force)
    plot(cache, ROOT / "fig_area.pdf")


if __name__ == "__main__":
    main()
