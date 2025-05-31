import csv
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# parser = argparse.ArgumentParser(description="Plot execution time breakdown from CSV file.")
# parser.add_argument('--fname', type=str, default="stallreason", help='Name of the CSV file without extension')
# parser.add_argument('--kernel', type=str, default="NTTPhase1_32", help='Kernel name to plot.')
# args = parser.parse_args()


dic_metricname_unit = {}

def format_kernel_name(kernel_name):
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
    sum_dram_bytes_read = 0
    for kernel_name, metrics in profiled_data.items():
        print(f"Kernel: {kernel_name}")
        for metric_name, values in metrics.items():
            # Convert to numpy array
            profiled_data[kernel_name][metric_name] = np.array(values)
            # Calculate mean and std
            mean_value = np.mean(profiled_data[kernel_name][metric_name])
            std_value = np.std(profiled_data[kernel_name][metric_name])
            sum_value = np.sum(profiled_data[kernel_name][metric_name])
            if metric_name == "dram__bytes_read.sum":
                sum_dram_bytes_read += sum_value
                # print(f"  {metric_name} : {sum_value:.2f} MBytes (mean: {mean_value:.2f}, std: {std_value:.2f})")
            print(f"  {metric_name} : {mean_value:.2f} {dic_metricname_unit[metric_name]} (std: {std_value:.2f})")
    print(f"Total DRAM Bytes Read: {sum_dram_bytes_read:.2f} MBytes")

def get_dram_sum(profiled_data):
    sum_dram_bytes_read = 0
    for kernel_name, metrics in profiled_data.items():
        for metric_name, values in metrics.items():
            if metric_name == "dram__bytes_read.sum":
                sum_dram_bytes_read += np.sum(values)
    return sum_dram_bytes_read


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
    
# plot_bar_scatter("l1tex__t_sector_hit_rate.pct", result, "L1 Cache Hit Rate(%)", f"{directory_path}/data/figure/{fname}-l1.pdf", cm.tab10.colors[0])
# plot_bar_scatter("lts__t_sector_hit_rate.pct", result, "L2 Cache Hit Rate(%)", f"{directory_path}/data/figure/{fname}-l2.pdf", cm.tab10.colors[2])
# plot_bar_scatter("lts__t_sector_op_read_hit_rate.pct", result, "L2 Cache Hit Rate(%)", f"{directory_path}/data/figure/{fname}-l2.pdf", cm.tab10.colors[2])


directory_path = os.path.dirname(os.path.abspath(__file__))

params = ["setB", "setC"]
optlevels = ["noopt", "reg", "l2", "shared"]


"""
penguin_means = {
    'NoOpt': (512, 1665),
    'Register': (362, 1229),
    'L2': (325, 1154),
    'SharedMem': (323, 1150),
}
"""

data = {}

for param in params:
    for optlevel in optlevels:
        fname_csv = f"{directory_path}/data/memaccess-{param}-{optlevel}-A4090.csv"
        if not os.path.exists(fname_csv):
            # print(f"File {fname_csv} does not exist. Skipping...")
            continue
        # print(f"Processing file: {fname_csv}")
        profiled_data = read_csv(fname_csv)
        dram_sum = get_dram_sum(profiled_data) / 1024
        print(f"Total DRAM Bytes Read for {param} with {optlevel}: {dram_sum:.2f} GBytes")
        if param not in data:
            data[param] = {}
        data[param][optlevel] = dram_sum

# Convert data to a format suitable for plotting
penguin_means = {
    'noopt': (data['setB']['noopt'], data['setC']['noopt']),
    'reg': (data['setB']['reg'], data['setC']['reg']),
    'l2': (data['setB']['l2'], data['setC']['l2']),
    'shared': (data['setB']['shared'], data['setC']['shared']),
}

x = np.array([0, 1.6])
width = 0.25  # the width of the bars
multiplier = 0

color_dict = {
    'noopt': cm.tab20b(2),
    'reg': cm.tab20b(6),
    'l2': cm.tab20b(10),
    'shared': cm.tab20b(14),
}
species = ("SetB", "SetC")
labels = ["NoOpt", "+Register", "+L2", "+SharedMem"]

fig, ax = plt.subplots(figsize=(9, 6))
for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, color=color_dict[attribute], label=labels[multiplier])
    ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=16)
    multiplier += 1
ax.set_ylabel('Total DRAM Access (GB)', fontsize=22)
ax.set_xticks(x + width * 1.5, species, fontsize=22)
ax.legend(loc='upper left', ncols=1, fontsize=18)
plt.tight_layout()
directory_path = os.path.dirname(os.path.abspath(__file__))
plt.savefig(f"{directory_path}/data/figure/opt_dram.pdf", dpi=500)
