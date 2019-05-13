import caffe
import numpy as np
import argparse
import os

original_model = "../models/vgg-16/deploy.prototxt"
original_weights = "../models/vgg-16/VGG_ILSVRC_16_layers.caffemodel"
obfuscated_model = "o_deploy.prototxt"

caffe.set_mode_cpu()
original_net = caffe.Net(original_model, 1, weights=original_weights)

# Enlarging an array: missing entries are filled with zeros
conv1_1_w = original_net.params['conv1_1'][0].data
conv1_1_b = original_net.params['conv1_1'][1].data
o_conv1_1_w = conv1_1_w.copy() # (64, 3, 3, 3)
o_conv1_1_b = conv1_1_b.copy() # (64,)
o_conv1_1_w.resize((256, 3, 3, 3))
o_conv1_1_b.resize((256,))
print o_conv1_1_w
print o_conv1_1_b

conv1_2_w = original_net.params['conv1_2'][0].data
conv1_2_b = original_net.params['conv1_2'][1].data
o_conv1_2_w = conv1_2_w.copy() # (64, 64, 3, 3)
o_conv1_2_b = conv1_2_b.copy() # (64,)
o_conv1_2_w.resize((256, 64, 3, 3))
o_conv1_2_b.resize((256,))
x = np.zeros((256, 192, 3, 3), dtype=float)
o_conv1_2_w = np.concatenate((o_conv1_2_w, x), axis=1) # (256, 256, 3, 3)
print o_conv1_2_w
print o_conv1_2_b

conv2_1_w = original_net.params['conv2_1'][0].data
conv2_1_b = original_net.params['conv2_1'][1].data
o_conv2_1_w = conv2_1_w.copy() # (128, 64, 3, 3)
o_conv2_1_b = conv2_1_b.copy() # (128,)
o_conv2_1_w.resize((256, 64, 3, 3))
o_conv2_1_b.resize((256,))
o_conv2_1_w = np.concatenate((o_conv2_1_w, x), axis=1) # (256, 256, 3, 3)
print o_conv2_1_w
print o_conv2_1_b

conv2_2_w = original_net.params['conv2_2'][0].data
conv2_2_b = original_net.params['conv2_2'][1].data
o_conv2_2_w = conv2_2_w.copy() # (128, 128, 3, 3)
o_conv2_2_b = conv2_2_b.copy() # (128,)
o_conv2_2_w.resize((256, 128, 3, 3))
o_conv2_2_b.resize((256,))
y = np.zeros((256, 128, 3, 3), dtype=float)
o_conv2_2_w = np.concatenate((o_conv2_2_w, y), axis=1) # (256, 256, 3, 3)
print o_conv2_2_w
print o_conv2_2_b

conv3_1_w = original_net.params['conv3_1'][0].data
conv3_1_b = original_net.params['conv3_1'][1].data
o_conv3_1_w = conv3_1_w.copy() # (256, 128, 3, 3)
o_conv3_1_b = conv3_1_b.copy() # (256,)
o_conv3_1_w.resize((256, 128, 3, 3))
o_conv3_1_b.resize((256,))
o_conv3_1_w = np.concatenate((o_conv3_1_w, y), axis=1) # (256, 256, 3, 3)
print o_conv3_1_w
print o_conv3_1_b

obfuscated_net = caffe.Net(obfuscated_model, 1, weights=original_weights)

obfuscated_net.params['o_conv1_1'][0].data[...] = o_conv1_1_w
obfuscated_net.params['o_conv1_1'][1].data[...] = o_conv1_1_b
obfuscated_net.params['o_conv1_2'][0].data[...] = o_conv1_2_w
obfuscated_net.params['o_conv1_2'][1].data[...] = o_conv1_2_b
obfuscated_net.params['o_conv2_1'][0].data[...] = o_conv2_1_w
obfuscated_net.params['o_conv2_1'][1].data[...] = o_conv2_1_b
obfuscated_net.params['o_conv2_2'][0].data[...] = o_conv2_2_w
obfuscated_net.params['o_conv2_2'][1].data[...] = o_conv2_2_b
obfuscated_net.params['o_conv3_1'][0].data[...] = o_conv3_1_w
obfuscated_net.params['o_conv3_1'][1].data[...] = o_conv3_1_b

obfuscated_net.save('O_VGG_ILSVRC_16_layers.caffemodel')
