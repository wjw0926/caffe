from collections import namedtuple
from matplotlib import pyplot as plt
import sys
import csv

inputfile = sys.argv[1]
Row = namedtuple('Row', ['slot', 'addr', 'time'])
threshold = 400

itcopy = []
oncopy = []
kernel = []

with open(inputfile, 'rb') as datafile:
    reader = csv.reader(datafile, delimiter=' ')
    rows = [Row(slot=int(row[0]), addr=int(row[1]), time=int(row[2])) for row in reader]

    for row in rows:
        if row.time < threshold:
            if row.addr in [0]:
                kernel.append(row)
            elif row.addr in [1]:
                itcopy.append(row)
            elif row.addr in [2]:
                oncopy.append(row)

    plt.plot([row.slot for row in itcopy],
             [row.time for row in itcopy],
             'bo', label='itcopy')
    plt.plot([row.slot for row in oncopy],
             [row.time for row in oncopy],
             'ro', label='oncopy')
    plt.plot([row.slot for row in kernel],
             [row.time for row in kernel],
             'g^', label='kernel')

    plt.xlabel('Time Slot Number')
    plt.ylabel('Probe Time (cycles)')
    plt.xlim(0, 200000)
    plt.ylim(0, 400)
    plt.legend()
    plt.show()
