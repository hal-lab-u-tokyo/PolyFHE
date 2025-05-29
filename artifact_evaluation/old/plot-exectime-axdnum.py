import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np

directory_path = "/opt/mount/PolyFHE"

dnums = [2, 3, 6, 18]
phantom_datas = [[0 for _ in range(len(dnums))] for _ in [15, 16]]
thiswork_datas = [[0 for _ in range(len(dnums))] for _ in [15, 16]] 
phantom_labels = [f"Phantom-logN{logn}" for logn in [15, 16]]
thiswork_labels = [f"Thiswork-logN{logn}" for logn in [15, 16]]
phantom_colors = ["lightblue", "blue"]
thiswork_colors = ["lightcoral", "red"]

# Read CSV
# CSV format
# logN,L,dnum,alpha,latency
def read_phantom():
    filenames = [f"profile/data/phantom/logN{logn}_dnum_halfL.csv" for logn in [15, 16]]
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

                if dnum not in dnums:
                    continue

                dnum_idx = dnums.index(dnum)
                data[dnum_idx] = latency

def read_thiswork():
    logns = [15, 16]
    for idx_logn in range(len(logns)):
        logn = logns[idx_logn]

        for idx_dnum in range(len(dnums)):
            dnum = dnums[idx_dnum]
            fname = f"profile/data/exectime/exectime-logN{logn}_L18_dnum{dnum}_SMemKB80.txt"
            filepath = os.path.join(directory_path, fname)
            data = thiswork_datas[idx_logn]

            if not os.path.exists(filepath):
                print(f"File {filepath} does not exist")
                continue

            with open(filepath) as f:
                # Find "Average time[us]: 194.25" and extract the value
                for line in f:
                    if "Average time[us]:" in line:
                        data[idx_dnum] = float(line.split(":")[1].strip())
                        break

def print_diff():
    # max diff
    max_improve = 0
    max_improve_logn = 0
    max_improve_dnum = 0
    max_overhead = 1
    max_overhead_logn = 0
    max_overhead_dnum = 0
    average_improve = 0
    for idx in range(len(phantom_datas)):
        for d_idx in range(len(dnums)):
            phantom_v = phantom_datas[idx][d_idx]
            thiswork_v = thiswork_datas[idx][d_idx]
            diff = (phantom_v - thiswork_v) / phantom_v * 100
            print(f"performance: {diff:.2f}")
            if diff > max_improve:
                max_improve = diff
                max_improve_logn = [15, 16][idx]
                max_improve_dnum = dnums[d_idx]
            elif diff < max_overhead:
                max_overhead = diff
                max_overhead_logn = [15, 16][idx]
                max_overhead_dnum = dnums[d_idx]
            average_improve += diff
    print(f"Max performance: {max_improve} at {max_improve_logn} with dnum {max_improve_dnum}")
    print(f"Min performance: {max_overhead} at {max_overhead_logn} with dnum {max_overhead_dnum}")
    print(f"Average: {average_improve / (len(phantom_datas) * len(dnums))}")

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
fig, ax = plt.subplots(figsize=(14, 8))
for idx in range(len(phantom_datas)):
    ax.plot(dnums, phantom_datas[idx], label=f"{phantom_labels[idx]}", marker=gen_marker(idx), color=phantom_colors[idx], markersize=10)

for idx in range(len(thiswork_datas)):
    ax.plot(dnums, thiswork_datas[idx], label=f"{thiswork_labels[idx]}", marker=gen_marker(idx), color=thiswork_colors[idx], markersize=10)

ax.set_xlabel("dnum", fontsize=38)
ax.set_ylabel("Latency [us]", fontsize=38)
ax.tick_params(axis='both', labelsize=28)
ax.legend(fontsize=28)
plt.savefig(f"{directory_path}/profile/figure/exectime-dnum.png", dpi=500, bbox_inches='tight', pad_inches=0)
print(f"Figure saved at {directory_path}/profile/figure/exectime-dnum.png")



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