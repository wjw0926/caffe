import caffe
import numpy as np
import argparse
import os

model = "../models/vgg-16/deploy.prototxt"
weights = "../models/vgg-16/VGG_ILSVRC_16_layers.caffemodel"

d_model = "decoy.prototxt"

caffe.set_mode_cpu()

original_net = caffe.Net(model, 1, weights=weights)

conv1_1_w = original_net.params['conv1_1'][0].data
conv1_1_b = original_net.params['conv1_1'][1].data
d_conv_w = conv1_1_w.copy() # (64, 3, 3, 3)
d_conv_b = conv1_1_b.copy() # (64,)

d_net = caffe.Net(d_model, 1, weights=weights)

d_net.params['d_conv'][0].data[...] = d_conv_w
d_net.params['d_conv'][1].data[...] = d_conv_b

d_net.save('decoy.caffemodel')
