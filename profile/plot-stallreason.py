import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np

# argument set1 or set2
if len(os.sys.argv) != 2:
    print("Usage: python plot-memtransfer.py <set1 or set2>")
    exit(1)
paramset = os.sys.argv[1]

directory_path = "/opt/mount/HiFive"
filename = [f"evalstall-stallreason-{w}-{paramset}.csv" for w in ["opt", "noopt", "phantom"]]
title = ["ThisWork", "ThisWork(Baseline)", "Phantom"]
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
weights = [{} for i in range(len(filename))]
for i in range(len(filename)):
    for j in range(len(metrics)):
        datas[i].append(0)

def format_name(name):
    # eliminate after "("
    name = name.split("(")[0]


def read_exectime():
    fnames = [f"evalstall-exectime-{i}-{paramset}.csv" for i in ["noopt", "opt", "phantom"]]
    
    for idx in range(len(fnames)):
        exectime = {}
        filepath = f"{directory_path}/profile/data/{fnames[idx]}"
        if not os.path.exists(filepath):
            print(f"File {filepath} does not exist")
            exit(1)

        with open(filepath) as f:
            inputs = csv.reader(f)
            l = [i for i in inputs]

            # CSV: Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
            time_idx = l[2].index("Time (%)")
            name_idx = l[2].index("Name")
            
            for i in range(3, len(l)):
                entry = l[i]
                if len(entry) == 0:
                    continue
                time = float(entry[time_idx])
                name = entry[name_idx]
                
                # format name
                name = format_name(name)

                if name not in exectime:
                    exectime[name] = time
                else:
                    exectime[name] += time
        
        total = sum(exectime.values())
        for key in exectime:
            weights[idx][key] = exectime[key] / total

def read_stallreason():
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
                metric_value = float(entry[metric_value_idx]) * weights[idx][kernel_name]

                # Sum up
                datas[idx][metric_idx] += metric_value


def filter_labels(data, labels):
    total = sum(data)
    return [label if (value / total) * 100 >= 10 else '' for label, value in zip(labels, data)]

def filter_autopct(pct):
    return f'{pct:.1f}%' if pct >= 10 else ''

read_exectime()
read_stallreason()

# Plot pie chart
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i in range(len(datas)):
    data = datas[i]
    ax = axes[i]
    ax.set_title(f"{title[i]}", fontsize=24)
    ax.pie(data, startangle=90, colors=cm.tab20.colors, labels=filter_labels(data, metrics), autopct=filter_autopct, textprops={'fontsize': 16})
    #ax.pie(data, startangle=90, colors=cm.tab20.colors, autopct=filter_autopct, textprops={'fontsize': 16})
    ax.axis('equal')


#plt.legend(metrics, fontsize=16, loc='best')
plt.tight_layout()
plt.savefig(f"{directory_path}/profile/figure/stallreason-{paramset}.eps", dpi=500, bbox_inches='tight', pad_inches=0)
print(f"Figure saved as {directory_path}/profile/figure/stallreason-{paramset}.eps")
plt.savefig(f"{directory_path}/profile/figure/stallreason-{paramset}.png", dpi=500, bbox_inches='tight', pad_inches=0)
print(f"Figure saved as {directory_path}/profile/figure/stallreason-{paramset}.png")

# Correspondence between metric name and color
#selected_label = []

#kernels = list(data.keys())
#reasons = list(next(iter(data.values())).keys())
#values = {reason: [data[kernel][reason] for kernel in kernels] for reason in reasons}
