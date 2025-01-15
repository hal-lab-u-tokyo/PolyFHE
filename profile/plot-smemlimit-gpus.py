import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np

directory_path = "/opt/mount/HiFive"

# Read log *.txt file

smem_limit = [i * 10 for i in range(1, 11)]
gpus = ["brisket", "iwashi", "rump"]
files = [["profile/data/smem_limit/" + str(gpus[i_gpus]) + "/" + str(i) + "0KB.txt" for i in range(1, 11)] for i_gpus in range(len(gpus))]
datas = [[0 for i in range(1, 11)] for i in range(len(gpus))]

for i_gpus in range(len(gpus)):
    print(f"GPU: {gpus[i_gpus]}")
    file_in_gpu = files[i_gpus]
    for idx in range(len(file_in_gpu)):
        fname = file_in_gpu[idx]
        filepath = os.path.join(directory_path, fname)

        if not os.path.exists(filepath):
            print(f"File {filepath} does not exist")
            continue

        with open(filepath) as f:
            # Find "Average time[us]: 194.25" and extract the value
            for line in f:
                if "Average time[us]:" in line:
                    datas[i_gpus][idx] = float(line.split(":")[1].strip())
                    break

print(smem_limit)
print(datas)

def gen_label(gpu):
    if gpu == "brisket":
        return "RTX 4090"
    elif gpu == "iwashi":
        return "A4000"
    elif gpu == "rump":
        return "A100"

# Plot
fig, ax = plt.subplots(figsize=(18, 10))
for i_gpus in range(len(gpus)):
    ax.plot(smem_limit, datas[i_gpus], marker='o', label=f'{gen_label(gpus[i_gpus])}', linewidth=2)

ax.set_xlabel('Limit Size of Shared Memory [KB]', fontsize=24)
ax.set_ylabel('Execution time[us]', fontsize=24)
# set font sizeof x and y axis
ax.tick_params(axis='both', labelsize=20)
ax.legend(fontsize=20)
plt.savefig(f"{directory_path}/profile/figure/smem_limit_gpus.png", dpi=500, bbox_inches='tight', pad_inches=0)
plt.savefig(f"{directory_path}/profile/figure/smem_limit_gpus.eps", dpi=500, bbox_inches='tight', pad_inches=0)
print(f"Figure saved at {directory_path}/profile/figure/smem_limit_gpus.png")