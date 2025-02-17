import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np

directory_path = "/opt/mount/PolyFHE"

limbs = [6 * i for i in range(1, 7)]
phantom_datas = [[0 for _ in range(len(limbs))] for _ in [15, 16]]
thiswork_datas = [[0 for _ in range(len(limbs))] for _ in [15, 16]] 
phantom_labels = [f"Phantom-logN{logn}" for logn in [15, 16]]
thiswork_labels = [f"Thiswork-logN{logn}" for logn in [15, 16]]
phantom_colors = ["lightblue", "blue"]
thiswork_colors = ["lightcoral", "red"]

# Read CSV
# CSV format
# logN,L,dnum,alpha,latency
def read_phantom():
    filenames = [f"profile/data/phantom/logN{logn}_L_dnum6.csv" for logn in [15, 16]]
    for idx in range(len(filenames)):
        fname = filenames[idx]
        data = phantom_datas[idx]
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

def read_thiswork():
    logns = [15, 16]
    for idx_logn in range(len(logns)):
        logn = logns[idx_logn]

        for idx_limb in range(len(limbs)):
            limb = limbs[idx_limb]
            fname = f"profile/data/exectime/exectime-logN{logn}_L{limb}_dnum3_SMemKB40.txt"
            filepath = os.path.join(directory_path, fname)
            data = thiswork_datas[idx_logn]

            if not os.path.exists(filepath):
                print(f"File {filepath} does not exist")
                continue

            with open(filepath) as f:
                # Find "Average time[us]: 194.25" and extract the value
                for line in f:
                    if "Average time[us]:" in line:
                        data[idx_limb] = float(line.split(":")[1].strip())
                        break

def print_diff():
    # max diff
    max_improve = 0
    max_improve_logn = 0
    max_improve_limb = 0
    max_overhead = 1
    max_overhead_logn = 0
    max_overhead_limb = 0
    average_improve = 0
    for idx in range(len(phantom_datas)):
        for l_idx in range(len(limbs)):
            phantom_v = phantom_datas[idx][l_idx]
            thiswork_v = thiswork_datas[idx][l_idx]
            diff = (phantom_v - thiswork_v) / phantom_v * 100
            print(f"Phantom: {phantom_v}, Thiswork: {thiswork_v}, Diff: {diff}")
            if diff > max_improve:
                max_improve = diff
                max_improve_logn = [15, 16][idx]
                max_improve_limb = limbs[l_idx]
            if diff < max_overhead:
                max_overhead = diff
                max_overhead_logn = [15, 16][idx]
                max_overhead_limb = limbs[l_idx]
    print(f"Max improve: {max_improve:.2f} (logN: {max_improve_logn}, limb: {max_improve_limb})")
    print(f"Max overhead: {max_overhead:.2f} (logN: {max_overhead_logn}, limb: {max_overhead_limb})")

read_phantom()
read_thiswork()
print("phantom_datas")
print(phantom_datas)
print("thiswork_datas")
print(thiswork_datas)

print_diff()

def gen_marker(idx):
    if idx == 0:
        return "o"
    elif idx == 1:
        return "v"
    
# Plot
fig, ax = plt.subplots(figsize=(18, 10))
for idx in range(len(phantom_datas)):
    ax.plot(limbs, phantom_datas[idx], label=f"{phantom_labels[idx]}", marker=gen_marker(idx), color=phantom_colors[idx], markersize=10)

for idx in range(len(thiswork_datas)):
    ax.plot(limbs, thiswork_datas[idx], label=f"{thiswork_labels[idx]}", marker=gen_marker(idx), color=thiswork_colors[idx], markersize=10)

ax.set_xlabel("Limbs", fontsize=24)
ax.set_ylabel("Latency [us]", fontsize=24)
ax.tick_params(axis='both', labelsize=20)
ax.legend(fontsize=20)
plt.savefig(f"{directory_path}/profile/figure/exectime-L.png", dpi=500,bbox_inches='tight', pad_inches=0)
print(f"Figure saved at {directory_path}/profile/figure/exectime-L.png")



"""
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
"""