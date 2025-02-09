import numpy as np
import matplotlib.pyplot as plt

"""
t_start,t_end,delta,smid,warpid
133606299618641,133606299618691,50,0,0
133606299618641,133606299618691,50,0,0
133606299618641,133606299618691,50,0,0
133606299618641,133606299618691,50,0,0
"""
np.set_printoptions(suppress=True,floatmode="fixed")

# Read CSV
file_path = "data/dram_latency.csv"
data = np.genfromtxt(file_path, delimiter=",", skip_header=1)

# print smid
smids = np.unique(data[:, 3])
print(f"smids={smids}")

# Extract smid = 0
target_smid = 47
data = data[data[:, 3] == target_smid]

# Check if (t_start,t_end) is identical in each warpid
warpids = np.unique(data[:, 4])
t_start_end = []
for warpid in warpids:
    warpid = int(warpid)
    mask = data[:, 4] == warpid
    t_starts = np.unique(data[mask, 0])
    t_ends = np.unique(data[mask, 1])
    if len(t_starts) != 1 or len(t_ends) != 1:
        print(f"Error: len(t_starts)={len(t_starts)}, len(t_ends)={len(t_ends)}")
        exit(1)
    t_start_end.append([t_starts[0], t_ends[0], t_ends[0] - t_starts[0]])


# Sort by t_start
t_start_end = sorted(t_start_end, key=lambda x: x[0])
print(t_start_end)

# Plot
fig, axs = plt.subplots(2)
fig.suptitle("DRAM Latency")

# Fig1: t_start and t_end
# y-axis: t_start, t_end
# x-axis: number of warps
n_warps = len(t_start_end)
axs[0].plot(range(n_warps), [x[0] for x in t_start_end], label="Start", marker="o", markersize=3)
axs[0].plot(range(n_warps), [x[1] for x in t_start_end], label="End", marker="x", markersize=3)
axs[0].set_ylabel("Clock")
axs[0].legend()

# Fig2: t_end - t_start
# y-axis: t_end - t_start
# x-axis: number of warps
axs[1].plot(range(n_warps), [x[2] for x in t_start_end], label="End - Start", marker="o", markersize=3)
axs[1].set_xlabel("Warp ID (sorted by start clock)")
axs[1].set_ylabel("Clock")
axs[1].legend()
plt.savefig("data/dram_latency.png", dpi=500)