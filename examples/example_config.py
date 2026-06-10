"""Example Ramulator2 configuration and simulation script"""

import ramulator

# Configure the simulation frontend that sends memory requests
frontend = ramulator.frontend.WindowTrace(
    clock_ratio=10,
    path="./examples/traces/trace.txt",
    bank=128,
    queue_len=32
)

# Instantiate the memory controller with the DRAM configuation
# ramulator.dram.DDR4(org_preset="DDR4_8Gb_x8", timing_preset="DDR4_2400R", rank=1),
ctrl = ramulator.controller.GenericDDR(
    dram=ramulator.dram.GDDR6(org_preset="GDDR6_8Gb_x8", timing_preset="GDDR6_2000_1350mV_double"),
    scheduler=ramulator.scheduler.FRFCFS(),
    #scope="Rank",
    read_buffer_size=32,
    refresh_manager=ramulator.refresh_manager.AllBank(),
    row_policy=ramulator.row_policy.Open(),
    addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
)
# Create a memory system with the controller
mem = ramulator.memory_system.GenericDRAM(
    clock_ratio=17,
    controllers=[ctrl] * 32,
    channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
)

# Run the simulation
sim = ramulator.Simulation(frontend, mem)
sim.run()

# sim.stats returns a nested Python dict of all simulation statistics
stats = sim.stats

# Guard here for `ramulator export`, which does not run the simulation 
# but only exports the config for pure C++ Ramulator library
# if stats:
#     # Controller stats are under memory_system → controller
#     ctrl_stats = stats["memory_system"]["controller"]

#     print(f"Controller cycles:     {ctrl_stats['cycles']}")
#     print(f"Avg read latency:      {ctrl_stats['avg_read_latency']:.1f} cycles")
#     print(f"Read requests:         {ctrl_stats['num_read_reqs']}")
#     print(f"Write requests:        {ctrl_stats['num_write_reqs']}")
#     print(f"Row hits:              {ctrl_stats['row_hits']}")
#     print(f"Row misses:            {ctrl_stats['row_misses']}")
#     print(f"Row conflicts:         {ctrl_stats['row_conflicts']}")

import sys
import json
print(json.dumps(stats, indent=2, ensure_ascii=False))