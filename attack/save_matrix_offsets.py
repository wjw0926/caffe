import re

inputfile = "dump-libopenblas.txt"

kernel = re.compile('^0000000000...... <sgemm_kernel>:')
itcopy = re.compile('^0000000000...... <sgemm_itcopy>:')
oncopy = re.compile('^0000000000...... <sgemm_oncopy>:')

def extract_offset(line):
    return line[10:16] + "\n"

result = open("offsets-libopenblas.txt", 'w')

with open(inputfile, 'rb') as f:
    while True:
        line = f.readline()
        if not line: break

        m = kernel.match(line)
        if m:
            result.write(extract_offset(line))
            continue

        m = itcopy.match(line)
        if m:
            result.write(extract_offset(line))
            continue

        m = oncopy.match(line)
        if m:
            result.write(extract_offset(line))
            continue

result.close()
print "Saving target offsets done"
