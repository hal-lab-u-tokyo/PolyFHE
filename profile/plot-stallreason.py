import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np

directory_path = "/opt/mount/HiFive"
filename = ["stallreason-noopt.csv", "stallreason-opt.csv", "stallreason-phantom.csv"]
title = ["ThisWork(Baseline)", "ThisWork", "Phantom"]
metrics = ["barrier", 
         "dispatch_stall",
         "drain",
         "imc_miss",
         "lg_throttle",
         "long_scoreboard",
         "math_pipe_throttle",
         "membar",
         "mio_throttle",
         "misc",
         "no_instruction",
         "not_selected",
         "selected",
         "short_scoreboard",
         "sleeping",
         "tex_throttle",
         "wait"]

datas = [[] for i in range(len(filename))]
for i in range(len(filename)):
    for j in range(len(metrics)):
        datas[i].append(0)
print(datas)

# Read CSV

for idx in range(len(filename)):
    fname = f"{directory_path}/profile/data/{filename[idx]}"

    if not os.path.exists(fname):
        print(f"File {fname} does not exist")
        exit(1)

    with open(fname) as f:
        inputs = csv.reader(f)
        l = [i for i in inputs]

        print(l[0])
        kernel_name_idx = l[0].index("Kernel Name")
        metric_name_idx = l[0].index("Metric Name")
        metric_value_idx = l[0].index("Metric Value")

        for i in range(1, len(l)):
            entry = l[i]
            # Kernel name
            # Remove "ckks::" prefix and after "(" 
            # Use up to 10 characters
            kernel_name = entry[kernel_name_idx]
            kernel_name = kernel_name.replace("ckks::", "")
            kernel_name = kernel_name.split("(")[0]
            #kernel_name = kernel_name[:15]

            # Metric name
            # Remove "smsp__warp_issue_stalled_" prefix and "_per_warp_active.pct" suffix
            metric_name = entry[metric_name_idx]
            metric_name = metric_name.replace("smsp__warp_issue_stalled_", "")
            metric_name = metric_name.replace("_per_warp_active.pct", "")

            if metric_name not in metrics:
                print(f"Unknown metric: {metric_name}")
                exit(1)
            metric_idx = metrics.index(metric_name)

            # Metric value
            metric_value = float(entry[metric_value_idx])

            # Sum up
            datas[idx][metric_idx] += metric_value


def filter_labels(data, labels):
    total = sum(data)
    return [label if (value / total) * 100 >= 20 else '' for label, value in zip(labels, data)]

def filter_autopct(pct):
    return f'{pct:.1f}%' if pct >= 20 else ''

# Plot pie chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i in range(len(datas)):
    data = datas[i]
    ax = axes[i]
    ax.set_title(f"{title[i]}", fontsize=24)
    ax.pie(data, startangle=90, colors=cm.tab20.colors, labels=filter_labels(data, metrics), autopct=filter_autopct, textprops={'fontsize': 16})
    ax.axis('equal')


#plt.xticks(rotation=60, fontsize=18, fontstyle='italic')
#plt.yticks(fontsize=18)
#plt.legend(reasons, fontsize=16, loc='best')
plt.tight_layout()
plt.savefig(f"{directory_path}/profile/figure/stallreason.png", dpi=500)
print(f"Figure saved as {directory_path}/profile/figure/stallreason.png")

# Correspondence between metric name and color
#selected_label = []

#kernels = list(data.keys())
#reasons = list(next(iter(data.values())).keys())
#values = {reason: [data[kernel][reason] for kernel in kernels] for reason in reasons}
