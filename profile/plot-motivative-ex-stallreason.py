import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np

directory_path = "/opt/mount/HiFive"
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

datas = [0 for i in range(len(metrics))]
weight = {}

def format_name(name):
    # eliminate after "("
    name = name.split("(")[0]
    """
    # if there is "phantom::", eliminate it
    name = name.split("phantom::")[-1]
    # if name contains "fnwt", name = ntt
    if "fnwt" in name:
        name = "NTT"
    elif "inwt" in name:
        name = "iNTT"
    elif "bconv"  in name:
        name = "BConv"
    elif "modup" in name:
        name = "BConv"
    elif "prod" in name:
        name = "Mult"
    elif "add" in name:
        name = "Add"
    """
    return name

def read_exectime():
    fname = "profile/data/phantom/phantom-L36.csv"
    exectime = {}
    filepath = os.path.join(directory_path, fname)
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist")
        exit(1)

    with open(filepath) as f:
        inputs = csv.reader(f)
        l = [i for i in inputs]

        # CSV: Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
        time_idx = l[0].index("Time (%)")
        name_idx = l[0].index("Name")
        
        for i in range(1, len(l)):
            entry = l[i]
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
        weight[key] = exectime[key] / total

def read_stallreason():
    filename = "profile/data/phantom/phantom-L36-stallreason.csv"
    filepath = os.path.join(directory_path, filename)
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
            kernel_name = entry[kernel_name_idx]
            kernel_name = format_name(kernel_name)
            w = weight[kernel_name]

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
            datas[metric_idx] += metric_value * w

read_exectime()
read_stallreason()
print(weight)
print(datas)

def filter_labels(data, labels):
    total = sum(data)
    return [label if (value / total) * 100 >= 20 else '' for label, value in zip(labels, data)]

def filter_autopct(pct):
    return f'{pct:.1f}%' if pct >= 5 else ''

# Plot pie chart
fig,ax = plt.subplots(figsize=(18, 10))
ax.pie(datas, startangle=90, colors=cm.tab20.colors, autopct=filter_autopct, textprops={'fontsize': 16})
#ax.pie(datas, startangle=90, colors=cm.tab20.colors, labels=filter_labels(datas, metrics), autopct=filter_autopct, textprops={'fontsize': 16})
ax.axis('equal')
#plt.legend(metrics, fontsize=16, loc='best')
plt.tight_layout()
plt.savefig(f"{directory_path}/profile/figure/motivative-ex-stallreason.png", dpi=500)
print(f"Figure saved as {directory_path}/profile/figure/motivative-ex-stallreason.png")