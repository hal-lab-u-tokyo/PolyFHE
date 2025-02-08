import numpy as np
import matplotlib.pyplot as plt

"""
t_start,t_end,delta,smid,warpid
133606299618641,133606299618691,50,0,0
133606299618641,133606299618691,50,0,0
133606299618641,133606299618691,50,0,0
133606299618641,133606299618691,50,0,0
"""

# Read CSV
file_path = "data/dram_latency.csv"
data = np.genfromtxt(file_path, delimiter=",", skip_header=1)

# Check if (t_start,t_end) is identical in each warpid
warpids = np.unique(data[:, 4])
t_start_end = []
for warpid in warpids:
    mask = data[:, 4] == warpid
    t_starts = np.unique(data[mask, 0])
    t_ends = np.unique(data[mask, 1])
    if len(t_starts) != 1 or len(t_ends) != 1:
        print(f"Error: warpid={warpid}")
        exit(1)
    t_start_end.append([t_starts[0], t_ends[0], t_ends[0] - t_starts[0]])


# Sort by t_start
t_start_end = sorted(t_start_end, key=lambda x: x[0])

# Plot

# two subplots
fig, axs = plt.subplots(2)
fig.suptitle("DRAM Latency")

# Fig1: t_start and t_end
# y-axis: t_start, t_end
# x-axis: number of warps
n_warps = len(t_start_end)
axs[0].plot(range(n_warps), [x[0] for x in t_start_end], label="Start")
axs[0].plot(range(n_warps), [x[1] for x in t_start_end], label="End")
axs[0].set_ylabel("Clock")
axs[0].legend()

# Fig2: t_end - t_start
# y-axis: t_end - t_start
# x-axis: number of warps
axs[1].plot(range(n_warps), [x[2] for x in t_start_end], label="End - Start")
axs[1].set_xlabel("Warp ID (sorted by start clock)")
axs[1].set_ylabel("Clock")
axs[1].legend()
plt.savefig("data/dram_latency.png", dpi=500)