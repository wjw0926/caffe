from collections import namedtuple
from matplotlib import pyplot as plt
import sys
import csv

inputfile = sys.argv[1]
Row = namedtuple('Row', ['slot', 'addr', 'time'])
threshold = 400

relu = []
conv = []
softmax = []
fc = []
pool = []

with open(inputfile, 'rb') as datafile:
    reader = csv.reader(datafile, delimiter=' ')
    rows = [Row(slot=int(row[0]), addr=int(row[1]), time=int(row[2])) for row in reader]

    for row in rows:
        if row.time < threshold:
            if row.addr in [0]:
                relu.append(row)
            elif row.addr in [1]:
                conv.append(row)
            elif row.addr in [2]:
                softmax.append(row)
            elif row.addr in [3]:
                fc.append(row)
            elif row.addr in [4]:
                pool.append(row)

    plt.plot([row.slot for row in relu],
             [row.time for row in relu],
             'k^', label='ReLU')
    plt.plot([row.slot for row in conv],
             [row.time for row in conv],
             'ro', label='Convolution')
    plt.plot([row.slot for row in softmax],
             [row.time for row in softmax],
             'kx', label='Softmax')
    plt.plot([row.slot for row in fc],
             [row.time for row in fc],
             'bo', label='InnerProduct')
    plt.plot([row.slot for row in pool],
             [row.time for row in pool],
             'go', label='Pooling')

    plt.xlabel('Time Slot Number')
    plt.ylabel('Probe Time (cycles)')
    plt.xlim(0, 100000)
    plt.ylim(0, 400)
    plt.legend()
    plt.show()
