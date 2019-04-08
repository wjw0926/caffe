#!/bin/bash

make
make pycaffe
cd build/lib/
objdump -d -j .text libcaffe.so.1.0.0 > dump-obfuscated.txt -j8
cd -
python save_offsets.py
