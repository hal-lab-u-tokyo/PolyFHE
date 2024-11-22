import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np

directory_path = "/opt/mount/HiFive"

data = {}

# Read CSV
# fname from argv[1]
if len(os.sys.argv) < 2:
    print("Usage: python3 plot-stallreason.py <name>")
    exit(1)
name = os.sys.argv[1]
fname = f"profile/data/{name}.csv"
filepath = os.path.join(directory_path, fname)

if not os.path.exists(filepath):
    print(f"File {filepath} does not exist")
    exit(1)

with open(filepath) as f:
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

        # Metric value
        metric_value = float(entry[metric_value_idx])

        # Sum up
        if kernel_name not in data:
            data[kernel_name] = {}
        if metric_name not in data[kernel_name]:
            data[kernel_name][metric_name] = metric_value
        else:
            data[kernel_name][metric_name] += metric_value


    # Normalize to 100%
    for kernel_name, metrics in data.items():
        sum_metrics = sum(metrics.values())
        #print(f"kernel:{kernel_name}, sum_metrics: {sum_metrics}")
        for metric_name, metric_value in metrics.items():
            data[kernel_name][metric_name] = metric_value * 100 /sum_metrics
            #print(f"\tmetric:{metric_name}, value: {metric_value}, normalized: {data[kernel_name][metric_name]}%")

    # Reorder metrics by value
    for kernel_name, metrics in data.items():
        print(kernel_name)
        data[kernel_name] = dict(sorted(metrics.items(), key=lambda item: item[1], reverse=True))

# Correspondence between metric name and color
all_labels = list(data[list(data.keys())[0]].keys())
all_colors = [cm.tab20.colors[i] for i in range(len(all_labels))]
selected_label = []

kernels = list(data.keys())
reasons = list(next(iter(data.values())).keys())
values = {reason: [data[kernel][reason] for kernel in kernels] for reason in reasons}


# Plot stacked bar chart
fig, ax = plt.subplots(figsize=(18, 10))

bottom = np.zeros(len(kernels))
for reason in reasons:
    k = kernels
    v = values[reason]
    c = all_colors[all_labels.index(reason)]
    ax.bar(k, v, color=c, bottom=bottom)
    bottom += v

plt.xticks(rotation=60, fontsize=18, fontstyle='italic')
plt.yticks(fontsize=18)
plt.legend(reasons, fontsize=16, loc='best')
plt.tight_layout()
plt.savefig(f"{directory_path}/profile/figure/{name}.png", dpi=500)

"""
for kernel in data:
    metrics = data[kernel]
    keys = list(metrics.keys())
    values = list(metrics.values())
    colors = [all_colors[all_labels.index(key)] for key in keys]
    ax.bar(kernel, values, bottom=0 ,color=colors)
    
    # If the metric is larger than 10% of sum, print label
    sum_metrics = sum(metrics.values())
    for metric_name, metric_value in metrics.items():
        if metric_value/sum_metrics > 0.5:
            if metric_name not in selected_label:
                selected_label.append(metric_name)
"""