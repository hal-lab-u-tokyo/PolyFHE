import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np

directory_path = "/opt/mount/HiFive"


# Read CSV
fname = "profile/data/phantom/phantom-L36.csv"
datas = {}

filepath = os.path.join(directory_path, fname)
if not os.path.exists(filepath):
    print(f"File {filepath} does not exist")
    exit(1)

with open(filepath) as f:
    inputs = csv.reader(f)
    l = [i for i in inputs]

    # CSV: Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
    time_idx = l[0].index("Time (%)")
    name_idx = l[0].index("Name")
    
    for i in range(1, len(l)):
        entry = l[i]
        time = float(entry[time_idx])
        name = entry[name_idx]
        
        # format name
        # eliminate after "("
        name = name.split("(")[0]
        # if there is "phantom::", eliminate it
        name = name.split("phantom::")[-1]
        # if name contains "fnwt", name = ntt
        if "fnwt" in name:
            name = "NTT"
        elif "inwt" in name:
            name = "iNTT"
        elif "bconv"  in name:
            name = "BConv"
        elif "modup" in name:
            name = "BConv"
        elif "prod" in name:
            name = "Mult"
        elif "add" in name:
            name = "Add"

        if name not in datas:
            datas[name] = time
        else:
            datas[name] += time

print(datas)

# Plot pie chart with font size 18
fig, ax = plt.subplots(figsize=(18, 10))
ax.pie(datas.values(), labels=datas.keys(), autopct='%1.1f%%', startangle=90, textprops={'fontsize': 18}, colors=cm.tab20.colors)
ax.axis('equal')
plt.savefig(f"{directory_path}/profile/figure/motivative-ex-time.png", dpi=500)
print(f"Figure saved at {directory_path}/profile/figure/motivative-ex-time.png")