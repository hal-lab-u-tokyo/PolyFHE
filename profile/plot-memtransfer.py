import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np

directory_path = "/opt/mount/HiFive"

data_opt = {}
data_noopt = {}
data_phantom = {}
datas = [data_opt, data_noopt, data_phantom]


# Read CSV
fname_opt = "profile/data/memtransfer-opt.csv"
fname_noopt = "profile/data/memtransfer-noopt.csv"
fname_phantom = "profile/data/memtransfer-phantom.csv"
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

            if metric_unit == "Mbyte":
                metric_value *= 1
            elif metric_unit == "Kbyte":
                metric_value *= 0.001
            elif metric_unit == "byte":
                metric_value /= 1000000
            else:
                print(f"Unknown unit {metric_unit}")
                exit(1)
            
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
metric = "dram__bytes_read.sum"
candidates = ["ThisWork", "ThisWork(Baseline)", "Phantom"]
num_iters = [6, 6, 5]
result = [datas[i][metric] / num_iters[i] for i in range(len(datas))]

print(metric)
print(candidates)
print(result)

ax.bar(candidates, result, color='tab:blue')
ax.set_ylabel("Data Transfer [MB]", fontsize=24)
ax.tick_params(axis='y', labelsize=24)
ax.tick_params(axis='x', labelsize=24)
plt.savefig(f"{directory_path}/profile/figure/memtransfer.png", dpi=500)
print(f"Figure saved at {directory_path}/profile/figure/memtransfer.png")