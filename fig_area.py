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

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
MEM_TXT = ROOT / "mem.txt"
RESULT_TXT = ROOT / "result_area.txt"
CACHE_FILE = ROOT / "fig_area_cache.json"

# ---- workload ----
BATCH = 4
HEAD = 32
DIM = 128
TOKEN_NUM = 2048
BITS = 4

# ---- hardware knobs ----
FREQ_HZ = 1e9
TILE_MAX = 128      # 优先把 tile_num 拉满，剩余预算再让 pe_num 长
PE_MAX = 1024

# ADKV Area: 3585.2712 um2
# Tender Area: 488.28 um2
# AxCore Area: 1056 um2

# Per-PE area (μm^2) at W4/FP16, 1 GHz — see file docstring.
PE_AREA_UM2 = {
    "ADKV":    3585.2712,   # AxCore-style approximate PE (W4 shared-add)
    "AxCore":  1056,
    # "KIVI":    2475,   # fpma
    # "KVQuant": 2475,
    # "Qserve":  2475,
    # "Atom":    2475,
    "Tender":  488.28,
    # "SKVQ":    2475,
}
DEFAULT_PE_AREA_UM2 = 2475

# ---- methods ----
METHODS = [
    # ("KIVI",    "gen_mem_trace_kivi.py",    []),
    # ("KVQuant", "gen_mem_trace_kvquant.py", []),
    ("AxCore",  "gen_mem_trace_kivi.py",    ["--group-size", "64"]),
    # ("Atom",    "gen_mem_trace_atom.py",    []),
    # ("Qserve",  "gen_mem_trace_qserve.py",  []),
    ("Tender",  "gen_mem_trace_qserve.py",    []),
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
MEMS = ["hbm3"]

# Area budgets (mm^2) — enough range to reach DRAM saturation.
#     0.005, 0.01, 0.02, 0.05, 0.1
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
    total = tile * pe
    return tile, pe, total * area_per_pe_mm2


def run_gen(script: str, pe_num: int, tile_num: int, extra: list) -> None:
    argv = [
        sys.executable, str(ROOT / script),
        "--pe-num", str(pe_num),
        "--tile-num", str(tile_num),
        "--batch", str(BATCH),
        "--head", str(HEAD),
        "--dim", str(DIM),
        "--token-num", str(TOKEN_NUM),
        "--bits", str(BITS),
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


def cache_key(mem: str, method: str, budget: float) -> str:
    return f"{mem}|{method}|{budget}"


def collect(force: bool) -> dict:
    cache = {}
    if CACHE_FILE.exists() and not force:
        cache = json.loads(CACHE_FILE.read_text())

    for mem in MEMS:
        for name, script, extra in METHODS:
            area_pp = PE_AREA_UM2.get(name, DEFAULT_PE_AREA_UM2)
            for budget in AREA_BUDGETS_MM2:
                key = cache_key(mem, name, budget)
                if key in cache:
                    continue
                sized = pick_pe_tile(budget, area_pp)
                if sized is None:
                    cache[key] = {"skipped": True}
                    CACHE_FILE.write_text(json.dumps(cache, indent=2))
                    continue
                tile, pe, area_used = sized
                print(f"[gen] mem={mem} method={name} budget={budget}"
                      f" -> tile={tile} pe={pe} area={area_used:.4f}mm^2")
                run_gen(script, pe_num=pe, tile_num=tile, extra=extra)
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


def tokens_per_s_per_mm2(entry: dict) -> float:
    return tokens_per_s(entry) / entry["area_used"]


# ---- plotting -----------------------------------------------------------
def plot(cache: dict, outpath: Path):
    fig, axes = plt.subplots(1, len(MEMS), figsize=(11, 4), sharey=False)
    if len(MEMS) == 1:
        axes = [axes]

    for ci, mem in enumerate(MEMS):
        ax = axes[ci]                # left  Y: absolute tokens/s
        ax2 = ax.twinx()             # right Y: tokens/s/mm^2

        for name, _, _ in METHODS:
            xs, ys_abs, ys_eff = [], [], []
            for budget in AREA_BUDGETS_MM2:
                entry = cache.get(cache_key(mem, name, budget))
                if not entry or entry.get("skipped"):
                    continue
                xs.append(entry["area_used"])
                ys_abs.append(tokens_per_s(entry))
                ys_eff.append(tokens_per_s_per_mm2(entry))
            if not xs:
                continue
            color = METHOD_COLORS.get(name, None)
            ax.plot(xs, ys_abs, marker="o", linewidth=1.8, markersize=4,
                    color=color, label=name)
            ax2.plot(xs, ys_eff, marker="s", linewidth=1.2, markersize=3,
                     linestyle="--", color=color, alpha=0.7)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax2.set_yscale("log")
        ax.set_xlabel("PE Area (mm$^2$)", fontsize=11)
        if ci == 0:
            ax.set_ylabel("Tokens / s  (solid, ─○)", fontsize=11)
        if ci == len(MEMS) - 1:
            ax2.set_ylabel("Tokens / s / mm$^2$  (dashed, ┅□)", fontsize=11)
        ax.set_title(mem.upper(), fontsize=11, fontweight="bold")
        ax.grid(True, which="both", linewidth=0.3, alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center",
               ncol=len(labels),
               bbox_to_anchor=(0.5, 1.0),
               frameon=False, fontsize=11,
               bbox_transform=fig.transFigure)

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
    plot(cache, ROOT / "fig_area.png")


if __name__ == "__main__":
    main()
