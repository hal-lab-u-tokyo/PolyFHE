import csv
import numpy as np

# csv_files = [f"./data/ncu-ntt12-A4090-{i}.csv" for i in [1, 2, 4, 6, 9, 12, 18, 36]]
csv_files = ["data/ncu-stallreason-evalkey-1.csv"]


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

                # Convert metric value to Kbytes
                if metric_unit == "byte":
                    metric_value /= 1024
                elif metric_unit == "Kbyte":
                    pass
                elif metric_unit == "Mbyte":
                    metric_value *= 1024
                elif metric_unit == "Gbyte":
                    metric_value *= 1024 * 1024

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
            print(f"  {metric_name} : {mean_value:.2f} (std: {std_value:.2f})")