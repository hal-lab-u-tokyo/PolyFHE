import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np

directory_path = os.path.dirname(os.path.abspath(__file__))
# fname = "stallreason-logN16_L36_K1_Q1480_brisket"
# fname_exectime = "exectime-logN16_L36_K1_Q1480_brisket"
fname = "stallreason-logN16_L36_K6_Q1580_brisket"
fname_exectime = "exectime-logN16_L36_K6_Q1580_brisket"

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

#datas = [0 for i in range(len(metrics))]

def format_name(name):
    # eliminate after "("
    name = name.split("(")[0]
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
    return name

def read_exectime(fname):
    filepath = os.path.join(directory_path, fname)
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist")
        exit(1)

    data = {}

    with open(filepath) as f:
        inputs = csv.reader(f)
        l = [i for i in inputs]
        
        idx = -1
        for i in range(len(l)):
            if "Time (%)" in l[i]:
                idx = i
                break
            idx += 1

        if idx == -1:
            print("No header found")
            exit(1)

        # CSV: Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
        time_idx = l[idx].index("Time (%)")
        name_idx = l[idx].index("Name")

        for i in range(idx + 1, len(l)):
            entry = l[i]
            
            if len(entry) < 1:
                break

            time = float(entry[time_idx])
            name = entry[name_idx]
            
            # format name
            name = format_name(name)

            if name not in data:
                data[name] = time
            else:
                data[name] += time
    return data        

def read_stallreason(fname, data_exectime):
    filepath = os.path.join(directory_path, fname)
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist")
        exit(1)

    data = {}

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
            kenrel_weight = data_exectime[kernel_name] / sum(data_exectime.values())

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
            metric_value = float(entry[metric_value_idx]) * kenrel_weight

            # Sum up
            if metric_name not in data:
                data[metric_name] = metric_value
            else:
                data[metric_name] += metric_value

    return data


def filter_labels(data, labels):
    total = sum(data)
    return [label if (value / total) * 100 >= 4.5 else '' for label, value in zip(labels, data)]

def filter_autopct(pct):
    return f'{pct:.1f}%' if pct >= 4.5 else ''

data_exectime = read_exectime(f"data/phantom/{fname_exectime}.csv")
data = read_stallreason(f"data/phantom/{fname}.csv", data_exectime)

print("exectime: ", data_exectime)
print("stall-reason: ", data)

fig,ax = plt.subplots(figsize=(10, 10))
ax.pie(data.values(), startangle=90, colors=cm.tab20.colors, labels=filter_labels(list(data.values()), metrics), autopct=filter_autopct, textprops={'fontsize': 28})
ax.axis('equal')
plt.tight_layout()
plt.savefig(f"{directory_path}/figure/{fname}.pdf", dpi=500, bbox_inches='tight', pad_inches=0)
print(f"Figure saved as {directory_path}/figure/{fname}.pdf")