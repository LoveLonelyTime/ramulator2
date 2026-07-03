# KIVI
python gen_mem_trace_kivi.py --pe-num 32 --tile-num 128 --batch 4 --token-num 2048 --bits 4 -o mem.txt
# KIVI 128g
python gen_mem_trace_kivi.py --pe-num 32 --tile-num 128 --group-size 128 --batch 4 --token-num 2048 --bits 4 -o mem.txt
# KVQuant
python gen_mem_trace_kvquant.py --pe-num 32 --tile-num 128 --batch 4 --token-num 2048 --bits 4 -o mem.txt
# Atom
python gen_mem_trace_atom.py --pe-num 32 --tile-num 128 --batch 4 --token-num 2048 --bits 4 -o mem.txt
# Qserve
python gen_mem_trace_qserve.py --pe-num 32 --tile-num 128 --batch 4 --token-num 2048 --bits 4 -o mem.txt
# ADKV
python gen_mem_trace_adkv.py --pe-num 32 --tile-num 128 --batch 4 --token-num 2048 --bits 4 -o mem.txt
# SKVQ
python gen_mem_trace_skvq.py --pe-num 32 --tile-num 128 --batch 4 --token-num 2048 --bits 4 -o mem.txt
# AxCore
python gen_mem_trace_kivi.py --pe-num 32 --tile-num 128 --group-size 64 --batch 4 --token-num 2048 --bits 4 -o mem.txt
# Tender
python gen_mem_trace_qserve.py --pe-num 32 --tile-num 128 --batch 4 --token-num 2048 --bits 4 -o mem.txt

# Ramulator
python examples/example_config.py > result.txt