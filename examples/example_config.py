"""Ramulator2 sim script parameterized by MEM env var.

Usage:  MEM=gddr6  python examples/example_config.py > result.txt
        MEM=hbm3   python examples/example_config.py > result.txt

Kept API-compatible with plot_speedup_grid.py; does not affect other users
who invoke it without setting MEM (defaults to gddr6 like before).
"""

import os
import json

import ramulator

MEM = os.environ.get("MEM", "gddr6").lower()

frontend = ramulator.frontend.WindowTrace(
    clock_ratio=10 if MEM == "gddr6" else 5,   # aim ~1 GHz frontend
    path="./mem.txt",
    bank=128,
    queue_len=256,
)

if MEM == "gddr6":
    ctrl = ramulator.controller.GenericDDR(
        dram=ramulator.dram.GDDR6(
            org_preset="GDDR6_8Gb_x8",
            timing_preset="GDDR6_2000_1350mV_double",
        ),
        scheduler=ramulator.scheduler.FRFCFS(),
        read_buffer_size=64,
        refresh_manager=ramulator.refresh_manager.AllBank(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
    )
    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=17,        # tCK 570 ps → 1.754 GHz mem, ~1.03 GHz fe
        controllers=[ctrl] * 32,
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
    )
elif MEM == "hbm3":
    ctrl = ramulator.controller.GenericDDR(
        dram=ramulator.dram.HBM3(
            org_preset="HBM3_4Gb",
            timing_preset="HBM3_6400Mbps",
        ),
        scheduler=ramulator.scheduler.FRFCFS(),
        read_buffer_size=64,
        refresh_manager=ramulator.refresh_manager.AllBank(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
    )
    # HBM3: 2 stacks × 16 pseudochannels = 32 controllers.
    # 之前用 16 controller 时 utilization 只 ~30–70%（受 controller 数上限限制，
    # 不是 row conflict），扩到 32 后 BW 直接跳到 900–1300 GB/s。
    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=8,         # tCK 625 ps → 1.6 GHz mem, exactly 1 GHz fe
        controllers=[ctrl] * 32,
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
    )
else:
    raise SystemExit(f"unknown MEM={MEM!r}, expected 'gddr6' or 'hbm3'")

sim = ramulator.Simulation(frontend, mem)
sim.run()

print(json.dumps(sim.stats, indent=2, ensure_ascii=False))
