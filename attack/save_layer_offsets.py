import re

inputfile = "dump-libcaffe.txt"

conv = re.compile('^0000000000...... <_\S+\d+ConvolutionLayer.....Forward_cpu')
relu = re.compile('^0000000000...... <_\S+\d+ReLULayer.....Forward_cpu')
pool = re.compile('^0000000000...... <_\S+\d+PoolingLayer.....Forward_cpu')
fc   = re.compile('^0000000000...... <_\S+\d+InnerProductLayer.....Forward_cpu')
softmax = re.compile('^0000000000...... <_\S+\d+SoftmaxLayer.....Forward_cpu')

def extract_offset(line):
    return line[10:16] + "\n"

result = open("offsets-libcaffe.txt", 'w')

with open(inputfile, 'rb') as f:
    while True:
        line = f.readline()
        if not line: break

        m = relu.match(line)
        if m:
            result.write(extract_offset(line))
            continue

        m = conv.match(line)
        if m:
            result.write(extract_offset(line))
            continue

        m = softmax.match(line)
        if m:
            result.write(extract_offset(line))
            continue

        m = fc.match(line)
        if m:
            result.write(extract_offset(line))
            continue

        m = pool.match(line)
        if m:
            result.write(extract_offset(line))
            continue

result.close()
print "Saving target offsets done"
