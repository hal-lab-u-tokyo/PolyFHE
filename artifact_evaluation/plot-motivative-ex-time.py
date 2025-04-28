import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np

directory_path = os.path.dirname(os.path.abspath(__file__))

# Read CSV
fname = "exectime-logN16_L36_K1_Q1480_brisket"
# fname = "exectime-logN16_L36_K6_Q1580_brisket"
fname_csv = f"data/phantom/{fname}.csv"

def read_data(fname_csv):
    filepath = os.path.join(directory_path, fname_csv)
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist")
        exit(1)

    res = {}
    with open(filepath) as f:
        inputs = csv.reader(f)
        l = [i for i in inputs]
        
        idx = -1
        for i in range(len(l)):
            if "Time (%)" in l[i]:
                idx = i
                break
            idx += 1

        if idx == -1:
            print("No header found")
            exit(1)

        # CSV: Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
        time_idx = l[idx].index("Time (%)")
        name_idx = l[idx].index("Name")
        
        for i in range(idx + 1, len(l)):
            entry = l[i]
            
            if len(entry) < 1:
                break

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

            if name not in res:
                res[name] = time
            else:
                res[name] += time
    return res

def filter_labels(data, labels):
    total = sum(data)
    return [label if (value / total) * 100 >= 5 else '' for label, value in zip(labels, data)]

def filter_autopct(pct):
    return f'{pct:.1f}%' if pct >= 5 else ''

tab10 = cm.tab10.colors
label_colors = {
    "NTT": tab10[0],
    "iNTT": tab10[9],
    "BConv": tab10[2],
    "Mult": tab10[1],
    "Add": tab10[8],
}

data = read_data(fname_csv)
colors = [label_colors[label] for label in data.keys()]
labels = data.keys()

# Plot pie chart with font size 18
fig, ax = plt.subplots(figsize=(10, 10))
ax.pie(data.values(), labels=filter_labels(data.values(), data.keys()), autopct=filter_autopct, startangle=90, textprops={'fontsize': 28}, colors=colors)
ax.axis('equal')

plt.savefig(f"{directory_path}/figure/{fname}.pdf", dpi=500, bbox_inches='tight', pad_inches=0)
print(f"Figure saved at {directory_path}/figure/{fname}.pdf")