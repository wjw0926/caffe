#!/bin/bash

make clean
make

CAFFE=~/caffe/build/lib/libcaffe.so.1.0.0
OFFSET=offsets-libcaffe.txt
SLEEP=$1

cd ~/caffe/
python test.py &
VICTIM_PID=$!
cd -

sleep ${SLEEP}

./attack ${CAFFE} ${OFFSET} 10000 &
ATTACK_PID=$!

wait ${VICTIM_PID}
wait ${ATTACK_PID}

python analyse_net.py result-10000.txt
