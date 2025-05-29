import csv
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

parser = argparse.ArgumentParser(description="Plot execution time breakdown from CSV file.")
parser.add_argument('--fname', type=str, default="stallreason", help='Name of the CSV file without extension')
parser.add_argument('--kernel', type=str, default="NTTPhase1_32", help='Kernel name to plot.')
args = parser.parse_args()

directory_path = os.path.dirname(os.path.abspath(__file__))
fname = args.fname
plot_kernel_name = args.kernel
fname_csv = f"{directory_path}/data/{fname}.csv"

dic_metricname_unit = {}

def format_kernel_name(kernel_name):
    return kernel_name.split("(")[0].strip()
    if "iNTTPhase1" in kernel_name:
        return "iNTTPhase1"
    elif "NTTPhase1" in kernel_name:
        return "NTTPhase1"
    elif "iNTTPhase2" in kernel_name:
        return "iNTTPhase2"
    elif "NTTPhase2" in kernel_name:
        return "NTTPhase2"
    elif "Mult" in kernel_name:
        return "Mult"
    elif "Add" in kernel_name:
        return "Add"
    elif "BConv" in kernel_name:
        return "BConv"
    else:
        return kernel_name.split("(")[0].strip()

def read_csv(file):
    profiled_data = {}
    with open(file, 'r') as f:
        reader = csv.reader(f)
        idx_kernel_name = 0
        idx_metric_name = 0
        idx_metric_value = 0
        for row in reader:
            if "ID" in row:
                # Header
                idx_kernel_name = row.index("Kernel Name")
                idx_metric_name = row.index("Metric Name")
                idx_metric_value = row.index("Metric Value")
                idx_metric_unit = row.index("Metric Unit")
            else:
                kernel_name = row[idx_kernel_name].split("(")[0].strip()
                metric_name = row[idx_metric_name]
                kernel_name = format_kernel_name(kernel_name)
                metric_value = float(row[idx_metric_value])
                metric_unit = row[idx_metric_unit]

                # Convert metric value to Mbytes
                if metric_unit == "byte":
                    metric_value /= 1024 * 1024
                elif metric_unit == "Kbyte":
                    metric_value /= 1024
                elif metric_unit == "Mbyte":
                    pass
                elif metric_unit == "Gbyte":
                    metric_value *= 1024
                elif metric_unit == "sector":
                    metric_value = (metric_value * 32) / (1024 * 1024)

                if metric_name not in dic_metricname_unit:
                    display_name = metric_unit
                    if metric_unit.endswith("byte"):
                        display_name = "MB"
                    elif metric_unit == "sector":
                        display_name = "MB"
                    dic_metricname_unit[metric_name] = display_name

                skip_list = ["max_rate", "ratio"]
                if any(skip in metric_name for skip in skip_list):
                    continue

                if kernel_name not in profiled_data:
                    profiled_data[kernel_name] = {}
                if metric_name not in profiled_data[kernel_name]:
                    profiled_data[kernel_name][metric_name] = []
                profiled_data[kernel_name][metric_name].append(metric_value)
    return profiled_data

def get_value(profiled_data, metric_name):
    kernels = []
    values = []
    kernel_names = sorted(profiled_data.keys())
    for kernel_name in kernel_names:
        v = profiled_data[kernel_name]
        for val in v[metric_name]:
            kernels.append(kernel_name)
            values.append(val)
    return kernels, values

def print_data(profiled_data):
    for kernel_name, metrics in profiled_data.items():
        print(f"Kernel: {kernel_name}")
        for metric_name, values in metrics.items():
            # Convert to numpy array
            profiled_data[kernel_name][metric_name] = np.array(values)
            # Calculate mean and std
            mean_value = np.mean(profiled_data[kernel_name][metric_name])
            std_value = np.std(profiled_data[kernel_name][metric_name])
            print(f"  {metric_name} : {mean_value:.2f} {dic_metricname_unit[metric_name]} (std: {std_value:.2f})")

result = read_csv(fname_csv)
print_data(result)

def plot_bar_scatter(metric_name, profiled_data, label, fname, color='tab:blue'):
    kernels, values = get_value(result, metric_name)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(kernels, values, label=metric_name, s=100, color=color, alpha=0.7)
    ax.set_ylabel(label, fontsize=24)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True)
    plt.savefig(fname, dpi=500, bbox_inches='tight', pad_inches=0)
    print(f"Figure saved at {fname}")
    
plot_bar_scatter("l1tex__t_sector_hit_rate.pct", result, "L1 Cache Hit Rate(%)", f"{directory_path}/data/figure/{fname}-l1.pdf", cm.tab10.colors[0])
plot_bar_scatter("lts__t_sector_hit_rate.pct", result, "L2 Cache Hit Rate(%)", f"{directory_path}/data/figure/{fname}-l2.pdf", cm.tab10.colors[2])
# plot_bar_scatter("lts__t_sector_op_read_hit_rate.pct", result, "L2 Cache Hit Rate(%)", f"{directory_path}/data/figure/{fname}-l2.pdf", cm.tab10.colors[2])