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
plot_name = os.path.splitext(filepath)[0]
file = open(filepath, "r")

for line in file:
     if re.search("Total", line):
         total.append(extract_float(line))
     if re.search("Validation", line):
         validation.append(extract_float(line))

print(get_plot_data(zip(total, validation), plot_name))
