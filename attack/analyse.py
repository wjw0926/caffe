from collections import namedtuple
from matplotlib import pyplot as plt
import sys
import csv

inputfile = sys.argv[1]
Row = namedtuple('Row', ['slot', 'addr', 'time'])
threshold = 500

relu = []
conv = []
lrn = []
dropout = []
softmax = []
fc = []
pool = []

with open(inputfile, 'rb') as datafile:
    reader = csv.reader(datafile, delimiter=' ')
    rows = [Row(slot=int(row[0]), addr=int(row[1]), time=int(row[2])) for row in reader]

    for row in rows:
        if row.time < threshold:
            if row.addr in [0,1]:
                relu.append(row)
            elif row.addr in [2,3]:
                conv.append(row)
            elif row.addr in [4,5]:
                lrn.append(row)
            elif row.addr in [6,7]:
                dropout.append(row)
            elif row.addr in [8,9]:
                softmax.append(row)
            elif row.addr in [10,11]:
                fc.append(row)
            elif row.addr in [12,13]:
                pool.append(row)

    plt.plot([row.slot for row in relu],
             [row.time for row in relu],
             'b^', label='ReLU')
    plt.plot([row.slot for row in conv],
             [row.time for row in conv],
             'bo', label='Convolution')
    plt.plot([row.slot for row in lrn],
             [row.time for row in lrn],
             'bx', label='LRN')
    plt.plot([row.slot for row in dropout],
             [row.time for row in dropout],
             'r^', label='Dropout')
    plt.plot([row.slot for row in softmax],
             [row.time for row in softmax],
             'rx', label='Softmax')
    plt.plot([row.slot for row in fc],
             [row.time for row in fc],
             'ro', label='InnerProduct')
    plt.plot([row.slot for row in pool],
             [row.time for row in pool],
             'go', label='Pooling')

    plt.xlabel('Time Slot Number')
    plt.ylabel('Probe Time (cycles)')
    plt.xlim(0, 30000)
    plt.ylim(0, 500)
    plt.legend()
    plt.show()
