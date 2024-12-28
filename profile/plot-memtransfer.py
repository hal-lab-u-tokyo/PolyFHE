import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np

directory_path = "/opt/mount/HiFive"

data_opt = {}
data_noopt = {}
count_opt = {}
count_noopt = {}

# Read CSV
fname_opt = "profile/data/memtransfer-opt.csv"
fname_noopt = "profile/data/memtransfer-noopt.csv"

for fname in [fname_opt, fname_noopt]:
    filepath = os.path.join(directory_path, fname)

    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist")
        exit(1)

    with open(filepath) as f:
        inputs = csv.reader(f)
        l = [i for i in inputs]

        metric_name_idx = l[0].index("Metric Name")
        metric_value_idx = l[0].index("Metric Value")

        if fname == fname_opt:
            data = data_opt
            count = count_opt
        else:
            data = data_noopt
            count = count_noopt

        for i in range(1, len(l)):
            entry = l[i]
            # Metric name
            metric_name = entry[metric_name_idx]
            if metric_name == "l1tex__t_sector_hit_rate.pct":
                metric_name = "L1 Cache Hit Rate"
            elif metric_name == "lts__t_sector_hit_rate.pct":
                metric_name = "LTS Cache Hit Rate"
            # Metric value
            metric_value = float(entry[metric_value_idx])

            
            if metric_name not in data:
                data[metric_name] = metric_value
                count[metric_name] = 1
            else:
                data[metric_name] += metric_value
                count[metric_name] += 1

        # Normalize
        for metric_name, metric_value in data.items():
            data[metric_name] = metric_value / count[metric_name]

print("Optimized")
print(data_opt)
print("Non-optimized")
print(data_noopt)

# Plot
fig, ax = plt.subplots(figsize=(18, 10))
metrics = list(data_opt.keys())
values = {"Optimized": [data_opt[metric] for metric in metrics], "Baseline": [data_noopt[metric] for metric in metrics]}

print(metrics)
print(values)

x = np.arange(len(metrics))
width = 0.25
multiplier = 0
for label, v in values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, v, width, label=label)
    ax.bar_label(rects, padding=2, fontsize=20)
    multiplier += 1

ax.legend(fontsize=24, loc='best')
ax.set_ylabel("Bytes", fontsize=24)
ax.tick_params(axis='y', labelsize=24)
ax.set_xticks(x + width/2, labels=metrics, fontsize=24)
ax.set_title("Optimization Impact on Global Memory Transfer", fontsize=24)
plt.savefig(f"{directory_path}/profile/figure/memtransfer.png", dpi=500)
print(f"Figure saved at {directory_path}/profile/figure/memtransfer.png")