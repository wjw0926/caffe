#!/bin/bash

make clean
make

BLAS=/opt/OpenBLAS/lib/libopenblas.so
OFFSET=offsets-libopenblas.txt
SLEEP=$1

cd ~/caffe/
python test.py &
VICTIM_PID=$!
cd -

sleep ${SLEEP}

./attack ${BLAS} ${OFFSET} 50 &
ATTACK_PID=$!

wait ${VICTIM_PID}
wait ${ATTACK_PID}

python analyse_mat.py result-50.txt
