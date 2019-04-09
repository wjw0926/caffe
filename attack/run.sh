#!/bin/bash

make clean
make

CAFFE=../build/lib/libcaffe.so.1.0.0
OFFSET=offsets.txt
CYCLE=$1

cd ..
python test.py &
cd -
VICTIM_PID=$!

sleep 2.1

./attack ${CAFFE} ${OFFSET} ${CYCLE} &
ATTACK_PID=$!

wait ${VICTIM_PID}
wait ${ATTACK_PID}
