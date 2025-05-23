import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np

if len(os.sys.argv) != 2:
    print("Usage: python plot-memtransfer.py <set1 or set2>")
    exit(1)
paramset = os.sys.argv[1]

directory_path = "/opt/mount/PolyFHE"


label = ["ThisWork", "ThisWork(Baseline)", "Phantom"]
metrics = ["L1 Cache Hit Rate", "L2 Cache Hit Rate"]
row_metrics = ["l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct"]

data_opt = {}
data_noopt = {}
data_phantom = {}
datas = [data_opt, data_noopt, data_phantom]

datas = {label[0]: [0, 0], label[1]: [0, 0], label[2]: [0, 0]}

# Read CSV
fname_opt = "profile/data/cachehit-opt.csv"
fname_noopt = "profile/data/cachehit-noopt.csv"
fname_phantom = "profile/data/cachehit-phantom.csv"
files = [fname_opt, fname_noopt, fname_phantom]

for idx in range(len(files)):
    fname = files[idx]
    filepath = os.path.join(directory_path, fname)

    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist")
        exit(1)

    with open(filepath) as f:
        inputs = csv.reader(f)
        l = [i for i in inputs]

        metric_name_idx = l[0].index("Metric Name")
        metric_value_idx = l[0].index("Metric Value")

        count = [0 for i in range(len(metrics))]
        for i in range(1, len(l)):
            entry = l[i]
            # Metric name
            metric_name = entry[metric_name_idx]
            metric_value = float(entry[metric_value_idx])

            if metric_name not in row_metrics:
                continue
            metric_idx = row_metrics.index(metric_name)
            datas[label[idx]][metric_idx] += metric_value
            count[metric_idx] += 1

        for i in range(len(metrics)):
            if count[i] == 0:
                continue
            datas[label[idx]][i] /= count[i]        

            
print(datas)

# Plot
fig, ax = plt.subplots(figsize=(14, 10))

x = np.arange(len(metrics))
width = 0.2
multiplier = 0
for attribute, measurement in datas.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1


ax.set_ylabel('Cache Hit Rate (%)', fontsize=24)
ax.set_ylim(0, 100)
ax.legend(loc='upper left', fontsize=14)
ax.set_xticks(x + width * 1.5, metrics, fontsize=18)
plt.savefig(f"{directory_path}/profile/figure/cachehit-{paramset}.png", dpi=500, bbox_inches='tight', pad_inches=0)
plt.savefig(f"{directory_path}/profile/figure/cachehit-{paramset}.eps", dpi=500, bbox_inches='tight', pad_inches=0)
print(f"Figure saved at {directory_path}/profile/figure/cachehit-{paramset}.png,eps")