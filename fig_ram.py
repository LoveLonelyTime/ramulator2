#!/usr/bin/env python3
"""
Attention-decode energy breakdown across quantization methods.

For each (memory, bits, method):
  - Run gen_mem_trace_*.py to obtain per-region byte counts (Q / K / PARAM /
    ZPSC / K_OUTLIER / K_OUTLIER_PTR ...) via the trailing statistics block.
  - DRAM energy per region = bytes * 8 * pJ_per_bit.
      HBM2  : 3.9 pJ/bit  (O'Connor et al., MICRO'17)
      GDDR5 : 14.0 pJ/bit (O'Connor et al., MICRO'17)
  - Compute energy = FMA_count * pJ_per_FMA(method).
      ADKV            : 1.5 pJ/FMA
      Tender / AxCore : 0.2 pJ/FMA
      others          : 1.0 pJ/FMA (tentative)
  - FMA count for one QK decode step = batch * head * token_num * dim.
    Each step generates `batch` output tokens, so
        Energy per token = total_pJ / batch.

Stacked bars: DRAM regions (bottom) + Compute (top). Two subplot rows
(HBM2, GDDR5) x two subplot cols (4-bit, 2-bit).

Cache: parsed byte counts saved to fig_ram_cache.json for fast re-runs.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import brokenaxes
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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
CACHE_FILE = ROOT / "fig_ram_cache.json"

# ---- workload ----
BATCH = 4
HEAD = 32
DIM = 128
TOKEN_NUM = 2048
PE_NUM = 32
TILE_NUM = 128

BITS_LIST = [4, 2]

# ---- methods (label, generator script, extra args) ----
METHODS = [
    ("KIVI",       "gen_mem_trace_kivi.py",    []),
    # ("KIVI 128g",  "gen_mem_trace_kivi.py",    ["--group-size", "128"]),
    ("KVQuant",    "gen_mem_trace_kvquant.py", []),
    ("AxCore",     "gen_mem_trace_kivi.py",    ["--group-size", "64"]),
    ("Atom",       "gen_mem_trace_atom.py",    []),
    ("Qserve",     "gen_mem_trace_qserve.py",  []),
    # ("SKVQ",       "gen_mem_trace_skvq.py",    []),
    # ("Tender",     "gen_mem_trace_qserve.py",  []),
    ("ADKV",       "gen_mem_trace_adkv.py",    []),
]
METHOD_LABELS = [m[0] for m in METHODS]

# ---- energy constants ----
PJ_PER_BIT = {"HBM2": 3.9, "GDDR5": 14.0}   # O'Connor MICRO'17
MEMORIES = ["GDDR5", "HBM2"]                # left GDDR5, right HBM2

DEFAULT_PJ_PER_FMA = 1.19 # 1GHz tcbn28hpcplusbwp7t40p140tt0p8v25c
PJ_PER_FMA_OVERRIDE = {
    "ADKV":   1.52, # 1GHz tcbn28hpcplusbwp7t40p140tt0p8v25c
    "AxCore": 0.11,
    "Tender": 0.26,
}

# ---- plotting ----
METADATA_REGIONS = {"ZPSC", "PARAM", "K_OUTLIER", "K_OUTLIER_PTR"}
SKIP_REGIONS = {"Q"}
REGION_ORDER = ["K", "Metadata"]
REGION_COLORS = {
    "K": "#4E79A7",
    "Metadata": "#7EA6D8",
    "Compute":  "#CEE1EF",
}
REG_LABELS = {
    "K": "KV Cache",
    "Metadata": "Metadata",
    "Compute":  "Compute",
}


def merge_metadata(regions: dict) -> dict:
    out = {}
    meta = 0
    for r, b in regions.items():
        if r in SKIP_REGIONS:
            continue
        if r in METADATA_REGIONS:
            meta += b
        else:
            out[r] = b
    if meta:
        out["Metadata"] = meta
    return out


# ---- helpers ------------------------------------------------------------
_REGION_LINE = re.compile(
    r"^\s*(\S+):\s+([\d,]+)\s+bursts,\s+([\d,]+)\s+bytes"
)


def parse_regions(text: str) -> dict:
    """Parse per-region byte counts from a gen script's stats block."""
    regions = {}
    for line in text.splitlines():
        m = _REGION_LINE.match(line)
        if not m:
            continue
        name = m.group(1).strip()
        byte_ = int(m.group(3).replace(",", ""))
        regions[name] = byte_
    return regions


def run_gen(script: str, bits: int, extra: list) -> dict:
    """Invoke a trace generator; return the parsed regions dict."""
    argv = [
        sys.executable, str(ROOT / script),
        "--pe-num", str(PE_NUM),
        "--tile-num", str(TILE_NUM),
        "--batch", str(BATCH),
        "--head", str(HEAD),
        "--dim", str(DIM),
        "--token-num", str(TOKEN_NUM),
        "--bits", str(bits),
        "-o", str(MEM_TXT),
    ] + list(extra)
    r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"cmd failed: {' '.join(argv)}")
    # Stats are printed to stdout when -o is set (see gen_mem_trace_*.py).
    return parse_regions(r.stdout + "\n" + r.stderr)


def cache_key(bits: int, method: str) -> str:
    return f"{bits}|{method}"


def collect_bytes() -> dict:
    """Gather per-region byte counts for all (bits, method) combos."""
    cache = {}
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())

    for bits in BITS_LIST:
        for name, script, extra in METHODS:
            key = cache_key(bits, name)
            if key in cache:
                continue
            print(f"[gen] bits={bits}  method={name}")
            cache[key] = run_gen(script, bits, extra)
            CACHE_FILE.write_text(json.dumps(cache, indent=2))
    return cache


def fma_count() -> int:
    """FMA count for one QK decode step."""
    return BATCH * HEAD * TOKEN_NUM * DIM


def pj_per_fma(method: str) -> float:
    return PJ_PER_FMA_OVERRIDE.get(method, DEFAULT_PJ_PER_FMA)


def build_energy(bytes_cache: dict) -> dict:
    """
    Returns:
      energy[mem][bits][method] = {
          "regions": {region: pJ per token},
          "compute": pJ per token,
          "total":   pJ per token,
      }
    """
    energy = {}
    fmas = fma_count()
    tokens = BATCH   # one decode step -> `batch` output tokens
    for mem in MEMORIES:
        pj_bit = PJ_PER_BIT[mem]
        energy[mem] = {}
        for bits in BITS_LIST:
            energy[mem][bits] = {}
            for name, _, _ in METHODS:
                regions = merge_metadata(bytes_cache[cache_key(bits, name)])
                region_pj = {r: b * 8 * pj_bit / tokens
                             for r, b in regions.items()}
                compute_pj = fmas * pj_per_fma(name) / tokens
                total = sum(region_pj.values()) + compute_pj
                energy[mem][bits][name] = {
                    "regions": region_pj,
                    "compute": compute_pj,
                    "total":   total,
                }
    return energy


# ---- plotting -----------------------------------------------------------
def plot(energy: dict, outpath: Path):
    """
    Two subplots (left GDDR5, right HBM2). Each subplot has two horizontal
    sections: all methods at 4-bit on the left, all methods at 2-bit on the
    right, separated by a small gap. Methods repeat within each section.
    """

    fig = plt.figure(figsize=(12, 4))
    sps = GridSpec(1, len(MEMORIES))
    axes = []

    for i in range(len(MEMORIES)):
        bax = brokenaxes.brokenaxes(
            ylims=((0, 10), (60, 100)),
            hspace=.05,
            d=0.005,
            subplot_spec=sps[i]
        )
        axes.append(bax)

    # union of regions across all cells, in preferred order
    seen = set()
    for mem in MEMORIES:
        for bits in BITS_LIST:
            for name in METHOD_LABELS:
                seen.update(energy[mem][bits][name]["regions"].keys())
    regions_used = [r for r in REGION_ORDER if r in seen]
    regions_used += [r for r in seen if r not in REGION_ORDER]

    n_methods  = len(METHOD_LABELS)
    section_gap = 1.5     # blank slots between the 4-bit and 2-bit sections
    bar_w = 0.75
    kivi_idx = METHOD_LABELS.index("KIVI")

    def section_xs(bi: int) -> np.ndarray:
        base = bi * (n_methods + section_gap)
        return np.arange(n_methods) + base

    def section_center(bi: int) -> float:
        base = bi * (n_methods + section_gap)
        return base + (n_methods - 1) / 2
    
    def fig_center() -> float:
        return n_methods - 1 + section_gap / 2 + 0.7

    legend_added = False
    print(", ".join(METHOD_LABELS))
    for ci, mem in enumerate(MEMORIES):
        print(f"{mem}:")
        ax = axes[ci]
        entries = energy[mem]

        # per-section KIVI totals used as denominators; each bit-width
        # normalizes independently.
        totals_abs  = np.zeros((n_methods, len(BITS_LIST)))
        totals_norm = np.zeros((n_methods, len(BITS_LIST)))

        for bi, bits in enumerate(BITS_LIST):
            xs = section_xs(bi)
            print(f"    {bits}:")

            kivi_total = (
                sum(entries[bits]["KIVI"]["regions"].values())
                + entries[bits]["KIVI"]["compute"]
            )
            scale = 100 / kivi_total if kivi_total > 0 else 100

            bottoms = np.zeros(n_methods)
            for reg in regions_used:
                heights = np.array([
                    entries[bits][m]["regions"].get(reg, 0.0) * scale
                    for m in METHOD_LABELS
                ])
                print(f"    {REG_LABELS[reg]}: {heights}")
                if heights.sum() == 0:
                    continue
                ax.bar(xs, heights, width=bar_w, bottom=bottoms,
                       color=REGION_COLORS.get(reg, "#aaaaaa"),
                       label=REG_LABELS[reg] if not legend_added else None)
                bottoms += heights



            heights = np.array([
                entries[bits][m]["compute"] * scale for m in METHOD_LABELS
            ])
            print(f"    {REG_LABELS["Compute"]}: {heights}")
            ax.bar(xs, heights, width=bar_w, bottom=bottoms,
                   color=REGION_COLORS["Compute"],
                   label="Compute" if not legend_added else None)

            totals_norm[:, bi] = bottoms + heights
            totals_abs[:, bi] = np.array([
                sum(entries[bits][m]["regions"].values()) + entries[bits][m]["compute"]
                for m in METHOD_LABELS
            ])

            for x, val in zip(xs, totals_norm[:, bi]):
                if val >= 90:
                    ax.text(x, val - 0.5, f"{val:.2f}", ha="center", va="top",
                            fontsize=9, fontweight="bold", rotation=90)
                else:
                    ax.text(x, val + 0.5, f"{val:.2f}", ha="center", va="bottom",
                            fontsize=9, fontweight="bold", rotation=90)

            legend_added = True

        y_top_est = 100

        ax.axs[1].tick_params(
            axis='y',
            labelleft=False
        )
        ax.axs[0].tick_params(axis="y", labelsize=11)

        # section labels ("4-bit" / "2-bit") above each half
        for bi, bits in enumerate(BITS_LIST):
            ax.axs[0].text(section_center(bi), 103,
                    f"{bits}-Bit",
                    ha="center", va="top", fontsize=11, fontweight="bold")

        # divider between the two sections
        divider_x = n_methods - 0.5 + section_gap / 2
        ax.axvline(divider_x, color="#888", linewidth=0.9,
                   linestyle="--", alpha=0.7)

        # title: report absolute KIVI energy per bit and ADKV reductions
        parts = []
        for bi, bits in enumerate(BITS_LIST):
            base = totals_abs[kivi_idx, bi]
            adkv = totals_abs[METHOD_LABELS.index("ADKV"), bi]
            parts.append(
                f"{bits}b KIVI={base/1e6:.1f}μJ, ADKV -{(base - adkv) / base * 100:.1f}%"
            )
        # title = (f"{mem}  ({PJ_PER_BIT[mem]} pJ/bit)   " + "   ".join(parts))
        # ax.set_title(title, fontsize=10)

        ax.axs[1].text(fig_center(), -10, f"{mem}",
            ha="center", va="top",
            fontsize=11, fontweight="bold",
        )

        # ax.set_ylim(0, y_top_est)
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
        all_xs = np.concatenate([section_xs(bi) for bi in range(len(BITS_LIST))])
        ax.axs[1].set_xticks(all_xs)
        ax.axs[1].set_xticklabels(METHOD_LABELS * len(BITS_LIST), rotation=30, ha="right", fontsize=11)
        ax.axs[0].set_yticks([60, 70, 80, 90, 100])
        ax.axs[0].set_yticklabels(["60%", "70%", "80%", "90%", "100%"])
        ax.axs[0].spines['top'].set_visible(True)
        ax.axs[0].spines['left'].set_visible(False)
        ax.axs[0].spines['right'].set_visible(False)
        ax.axs[1].spines['left'].set_visible(False)
        ax.axs[1].spines['right'].set_visible(False)
    
    axes[0].set_ylabel("Energy Breakdown (%)", fontsize=11)
    axes[1].axs[0].yaxis.set_tick_params(labelleft=False)


    handles, labels = axes[0].axs[0].get_legend_handles_labels()
    plt.figlegend(handles, labels,
           loc='lower center',
           bbox_to_anchor=(0.5, 0.92),
           ncol=4,
           frameon=False,
           fontsize=11,
           bbox_transform=fig.transFigure)

    # fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    print(f"saved {outpath}")


# ---- main ---------------------------------------------------------------
def main():
    bytes_cache = collect_bytes()
    energy = build_energy(bytes_cache)
    plot(energy, ROOT / "fig_ram.pdf")


if __name__ == "__main__":
    main()
