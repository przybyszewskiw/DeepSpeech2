import re
import sys
import os

total = []
validation = []


def extract_float(line):
    x = re.findall("\d+\.\d+", line)
    return float(x[0])

def get_plot_data(data, plot_name):
    plot  = """\pgfplotstableread[row sep=\\\\,col sep=&]{
    epochs & training & validation \\\\ """
    for i, (training, validation) in enumerate(data):
        plot += "\n   " + str(i + 1) + " & " + ("%.2f" % training) + " & " + ("%.2f" % validation)+ " \\\\"
    plot += "\n}\\" + plot_name
    return plot


filepath = sys.argv[1]
if len(sys.argv) > 2:
    gpus = int(sys.argv[2])
else:
    gpus = 1
plot_name = os.path.splitext(filepath)[0]
file = open(filepath, "r")

sum_tot = 0.0
sum_val = 0.0
i_tot = 0
i_val = 0
for line in file:
     if re.search("Total", line):
         i_tot += 1
         sum_tot += extract_float(line)
         if i_tot % gpus == 0:
             total.append(sum_tot / gpus)
             sum_tot = 0.0
     if re.search("Validation", line):
         i_val += 1
         sum_val += extract_float(line)
         if i_val % gpus == 0:
             validation.append(sum_val / gpus)
             sum_val = 0.0

print(get_plot_data(zip(total, validation), plot_name))
