import csv
import numpy as np

# csv_files = [f"./data/ncu-ntt12-A4090-{i}.csv" for i in [1, 2, 4, 6, 9, 12, 18, 36]]
csv_files = [f"data/ncu-v25-{i}.csv" for i in [0, 1]]

dic_metricname_unit = {}

for file in csv_files:
    profiled_data = {}

    print("=========================")
    print(f"{file}")
    print("=========================")

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

    for kernel_name, metrics in profiled_data.items():
        print(f"Kernel: {kernel_name}")
        for metric_name, values in metrics.items():
            # Convert to numpy array
            profiled_data[kernel_name][metric_name] = np.array(values)
            # Calculate mean and std
            mean_value = np.mean(profiled_data[kernel_name][metric_name])
            std_value = np.std(profiled_data[kernel_name][metric_name])
            print(f"  {metric_name} : {mean_value:.2f} {dic_metricname_unit[metric_name]} (std: {std_value:.2f})")