import matplotlib.pyplot as plt
import numpy as np

directory_path = "/opt/mount/HiFive"

benchmarks = ("Sobel Filter", "Linear Regression", "Multi-Layer Perceptron")

works = {
    'This Work(GPU)': (189.95, 195.82, 217.19),
    'This Work(Baseline)': (189.95, 195.82, 217.19),
}

x = np.arange(len(benchmarks))
width = 0.2
multiplier = 0

fig, ax = plt.subplots(figsize=(12, 8))

for attribute, measurement in works.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Execution Time (ms)', fontsize=24)
ax.set_xticks(x + width * 1.5, benchmarks, fontsize=18)
ax.legend(loc='upper left', ncols=2, fontsize=14)
ax.set_ylim(0, 300)

plt.savefig(f"{directory_path}/profile/figure/e2e-exectime.png", dpi=500)
print(f"Figure saved at {directory_path}/profile/figure/e2e-exectime.png")