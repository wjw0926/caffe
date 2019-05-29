#!/bin/bash

make -j8
make pycaffe
python test.py
