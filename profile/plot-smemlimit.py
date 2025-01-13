import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np

directory_path = "/opt/mount/HiFive"

# Read log *.txt file

smem_limit = [i * 10 for i in range(1, 11)]
files = ["profile/data/smem_limit/" + str(i) + "0KB.txt" for i in range(1, 11)]
datas = [0 for i in range(1, 11)]

for idx in range(len(files)):
    fname = files[idx]
    filepath = os.path.join(directory_path, fname)

    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist")
        exit(1)

    with open(filepath) as f:
        # Find "Average time[us]: 194.25" and extract the value
        for line in f:
            if "Average time[us]:" in line:
                datas[idx] = float(line.split(":")[1].strip())
                break

print(smem_limit)
print(datas)

# Plot
fig, ax = plt.subplots(figsize=(18, 10))
ax.plot(smem_limit, datas, marker='o', label='Average time [us]')
ax.set_xlabel('Limit Size of Shared Memory [KB]', fontsize=24)
ax.set_ylabel('Average time[us]', fontsize=24)
# set font sizeof x and y axis
ax.tick_params(axis='both', labelsize=20)

plt.savefig(f"{directory_path}/profile/figure/smem_limit.png", dpi=500)
print(f"Figure saved at {directory_path}/profile/figure/smem_limit.png")