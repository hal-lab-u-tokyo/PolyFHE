import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np

directory_path = "/opt/mount/HiFive"

filenames = [f"profile/data/phantom/logN{logn}_L_dnum{dnum}.csv" for logn in [15, 16] for dnum in [3,6]]
labels = [f"logN={logn}, dnum={dnum}" for logn in [15, 16] for dnum in [3,6]]
limbs = [6 * i for i in range(1, 7)]
datas = [[0 for _ in range(len(limbs))] for _ in range(len(filenames))]

# Read CSV
# CSV format
# logN,L,dnum,alpha,latency

for idx in range(len(filenames)):
    fname = filenames[idx]
    data = datas[idx]
    filepath = os.path.join(directory_path, fname)

    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist")
        exit(1)

    with open(filepath) as f:
        print(f"Reading {filepath}")
        inputs = csv.reader(f)
        l = [i for i in inputs]
        for i in range(1, len(l)):
            entry = l[i]
            logN = int(entry[0])
            limb = int(entry[1])
            dnum = int(entry[2])
            alpha = int(entry[3])
            latency = float(entry[4])

            limb_idx = limbs.index(limb)
            data[limb_idx] = latency

print("datas")
print(datas)
print("limbs")
print(limbs)

# Plot
fig, ax = plt.subplots(figsize=(18, 10))
for idx in range(len(filenames)):
    ax.plot(limbs, datas[idx], label=f"{labels[idx]}", marker='o')

ax.set_xlabel("Limbs", fontsize=24)
ax.set_ylabel("Execution Time [us]", fontsize=24)
ax.tick_params(axis='both', labelsize=20)
ax.legend(fontsize=20)
plt.savefig(f"{directory_path}/profile/figure/exectime-axL.png", dpi=500)
print(f"Figure saved at {directory_path}/profile/figure/exectime-axL.png")