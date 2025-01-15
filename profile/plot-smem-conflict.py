import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np

# argument set1 or set2
if len(os.sys.argv) != 2:
    print("Usage: python plot-smem-conflict.py <set1 or set2>")
    exit(1)
paramset = os.sys.argv[1]


directory_path = "/opt/mount/HiFive"

data_opt = {}
data_noopt = {}
data_phantom = {}
datas = [data_opt, data_noopt, data_phantom]


# Read CSV
fname_opt = f"profile/data/smem-conflict-opt-{paramset}.csv"
fname_noopt = f"profile/data/smem-conflict-noopt-{paramset}.csv"
fname_phantom = f"profile/data/smem-conflict-phantom-{paramset}.csv"
files = [fname_opt, fname_noopt, fname_phantom]

for idx in range(len(files)):
    fname = files[idx]
    data = datas[idx]
    filepath = os.path.join(directory_path, fname)

    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist")
        exit(1)

    with open(filepath) as f:
        inputs = csv.reader(f)
        l = [i for i in inputs]

        metric_name_idx = l[0].index("Metric Name")
        metric_value_idx = l[0].index("Metric Value")
        metric_unit_idx = l[0].index("Metric Unit")

        for i in range(1, len(l)):
            entry = l[i]
            # Metric name
            metric_name = entry[metric_name_idx]
            metric_value = float(entry[metric_value_idx])
            metric_unit = entry[metric_unit_idx]

            if metric_name not in data:
                data[metric_name] = metric_value
            else:
                data[metric_name] += metric_value


print("Optimized")
print(datas[0])
print("Non-optimized")
print(datas[1])
print("Phantom")
print(datas[2])

# Plot
fig, ax = plt.subplots(figsize=(18, 10))
metric = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
candidates = ["ThisWork", "ThisWork(Baseline)", "Phantom"]
num_iters = [6, 6, 5]
result = [datas[i][metric] / num_iters[i] for i in range(len(datas))]

# normalize to ThisWork
result = [result[i] / result[0] for i in range(len(result))]

print(metric)
print(candidates)
print(result)

print(f"(Phantom - ThisWork) / Phantom = {(result[2] - result[0]) / result[2] * 100:.2f}%")
print(f"(Baseline - ThisWork) / Baseline = {(result[1] - result[0]) / result[1] * 100:.2f}%")

ax.bar(candidates, result, color='tab:blue')
ax.set_ylabel("Shared Memory Conflict Count (Normalized)", fontsize=24)
ax.tick_params(axis='y', labelsize=24)
ax.tick_params(axis='x', labelsize=24)
plt.savefig(f"{directory_path}/profile/figure/smem-conflict-{paramset}.eps", dpi=500,bbox_inches='tight', pad_inches=0)
plt.savefig(f"{directory_path}/profile/figure/smem-conflict-{paramset}.png", dpi=500,bbox_inches='tight', pad_inches=0)
print(f"Figure saved at {directory_path}/profile/figure/smem-conflict-{paramset}.eps")