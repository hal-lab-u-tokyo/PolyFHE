
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm

species = ("SetB", "SetC")
penguin_means = {
    'NoOpt': (512, 1665),
    '+Register': (362, 1229),
    '+L2': (325, 1154),
    '+SharedMem': (323, 1150),
}

# x = np.arange(len(species))  # the label locations
x = np.array([0, 1.6])
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(9, 6))

color_dict = {
    'NoOpt': cm.tab20b(2),
    '+Register': cm.tab20b(6),
    '+L2': cm.tab20b(10),
    '+SharedMem': cm.tab20b(14),
}

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=color_dict[attribute])
    ax.bar_label(rects, padding=3, fontsize=16)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Execution Time (us)', fontsize=22)
# ax.set_title('Penguin attributes by species')
ax.set_xticks(x + width * 1.5, species, fontsize=22)
ax.legend(loc='upper left', ncols=1, fontsize=18)
# ax.set_ylim(0, 250)

directory_path = os.path.dirname(os.path.abspath(__file__))
plt.savefig(f"{directory_path}/data/figure/opt_exectime.pdf", dpi=500)