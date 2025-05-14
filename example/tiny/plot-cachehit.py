import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np

param_size = "large"
directory_path = os.path.dirname(os.path.abspath(__file__)) + "/data/" + param_size + "/"

# label = ["Register", "L2", "NoOpt"]
label = ["L2", "NoOpt"]
metrics = ["L1 Cache Hit Rate", "L2 Cache Hit Rate"]
row_metrics = ["l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct"]

datas = {l : [0, 0] for l in label}

# Read CSV
# fname_register = directory_path + "cachehit-reg.csv"
fname_l2 = directory_path + "cachehit-l2.csv"
fname_noopt = directory_path + "cachehit-noopt.csv"
# files = [fname_register, fname_l2, fname_noopt]
files = [fname_l2, fname_noopt]

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
width = 0.3
multiplier = 0
colors = ["darkblue", "royalblue", "lightsteelblue"]
for attribute, measurement in datas.items():
    offset = width * multiplier
    print("x:", x)
    print("offset:", offset)
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[multiplier])
    ax.bar_label(rects, padding=3, fontsize=24)
    multiplier += 1


ax.set_ylabel('Cache Hit Rate (%)', fontsize=24)
ax.set_ylim(0, 100)
ax.legend(loc='upper left', fontsize=20)
ax.set_xticks(x + width, metrics, fontsize=24)
ax.tick_params(axis='y', labelsize=24)
outname = directory_path + "cachehit"
plt.savefig(f"{outname}.png", dpi=500, bbox_inches='tight', pad_inches=0)
print(f"Figure saved at {outname}.png")