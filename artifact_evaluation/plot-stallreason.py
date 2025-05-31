import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Plot execution time breakdown from CSV file.")
parser.add_argument('--fname', type=str, default="stallreason", help='Name of the CSV file without extension')
parser.add_argument('--kernel', type=str, default="NTTPhase1_32", help='Kernel name to plot.')
args = parser.parse_args()

directory_path = os.path.dirname(os.path.abspath(__file__))
fname = args.fname
plot_kernel_name = args.kernel
fname_csv = f"{directory_path}/data/{fname}.csv"
title = "Stall Reason Breakdown"

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

data = {}

def format_name(name):
    # eliminate after "("
    tmp =  name.split("(")[0]
    return tmp



def read_stallreason(fname):
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
            kernel_name = entry[kernel_name_idx]
            kernel_name = format_name(kernel_name)

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
            # print(f"Kernel: {kernel_name}, Metric: {metric_name}, Value: {metric_value}")

            # Sum up
            if kernel_name not in data:
                data[kernel_name] = {}
            else:
                if metric_name in data[kernel_name]:
                    data[kernel_name][metric_name].append(metric_value)
                else:
                    data[kernel_name][metric_name] = [metric_value]



read_stallreason(fname_csv)
avg_data = {}

# print data
for i, (kernel_name, metrics_data) in enumerate(data.items()):
    print(f"Kernel: {kernel_name}")
    for metric in metrics:
        if metric in metrics_data:
            avg = np.mean(metrics_data[metric])
            std = np.std(metrics_data[metric])
            if (avg != 0) and (std / avg > 0.3):
                # If standard deviation is more than 30% of the average, print a warning
                print(f"== Warning: {metric} has high standard deviation: {std:.2f} ==")
            print(f"  {metric}: {avg:.2f} ± {std:.2f}")

            if kernel_name == plot_kernel_name:
                avg_data[metric] = avg
        else:
            print(f"  {metric}: 0.0")

# Plot pie chart
def filter_labels(data, labels):
    return [label if value >= 5 else '' for label, value in zip(labels, data)]

def filter_autopct(pct):
    return f'{pct:.1f}%' if pct >= 6 else ''

tab10 = cm.tab10.colors
label_colors = {
    "NTT": tab10[0],
    "iNTT": tab10[9],
    "BConv": tab10[2],
    "Mult": tab10[1],
    "Add": tab10[8],
}

# colors = [label_colors[label] for label in data.keys()]

fig, ax = plt.subplots(figsize=(10, 10))
ax.pie(avg_data.values(),
        labels=filter_labels(avg_data.values(), avg_data.keys()),
        autopct=filter_autopct,
        startangle=90,
        colors=cm.tab20.colors,
        textprops={'fontsize': 28})
ax.axis('equal')

plt.savefig(f"{directory_path}/data/figure/{fname}-{plot_kernel_name}.pdf", dpi=500, bbox_inches='tight', pad_inches=0)
print(f"Figure saved at {directory_path}/data/figure/{fname}-{plot_kernel_name}.pdf")