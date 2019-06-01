import caffe
import numpy as np
import argparse
import os

model = "../models/vgg-16/deploy.prototxt"
weights = "../models/vgg-16/VGG_ILSVRC_16_layers.caffemodel"
o_model = "o_deploy_baseline.prototxt"

caffe.set_mode_cpu()
original_net = caffe.Net(model, 1, weights=weights)

# Enlarging an array: missing entries are filled with zeros
conv1_1_w = original_net.params['conv1_1'][0].data
conv1_1_b = original_net.params['conv1_1'][1].data
o_conv1_1_w = conv1_1_w.copy() # (64, 3, 3, 3)
o_conv1_1_b = conv1_1_b.copy() # (64,)
o_conv1_1_w.resize((512, 3, 3, 3))
o_conv1_1_b.resize((512,))

conv1_2_w = original_net.params['conv1_2'][0].data
conv1_2_b = original_net.params['conv1_2'][1].data
o_conv1_2_w = conv1_2_w.copy() # (64, 64, 3, 3)
o_conv1_2_b = conv1_2_b.copy() # (64,)
o_conv1_2_w.resize((512, 64, 3, 3))
o_conv1_2_b.resize((512,))
x = np.zeros((512, 448, 3, 3), dtype=float)
o_conv1_2_w = np.concatenate((o_conv1_2_w, x), axis=1) # (512, 512, 3, 3)

conv2_1_w = original_net.params['conv2_1'][0].data
conv2_1_b = original_net.params['conv2_1'][1].data
o_conv2_1_w = conv2_1_w.copy() # (128, 64, 3, 3)
o_conv2_1_b = conv2_1_b.copy() # (128,)
o_conv2_1_w.resize((512, 64, 3, 3))
o_conv2_1_b.resize((512,))
o_conv2_1_w = np.concatenate((o_conv2_1_w, x), axis=1) # (512, 512, 3, 3)

conv2_2_w = original_net.params['conv2_2'][0].data
conv2_2_b = original_net.params['conv2_2'][1].data
o_conv2_2_w = conv2_2_w.copy() # (128, 128, 3, 3)
o_conv2_2_b = conv2_2_b.copy() # (128,)
o_conv2_2_w.resize((512, 128, 3, 3))
o_conv2_2_b.resize((512,))
y = np.zeros((512, 384, 3, 3), dtype=float)
o_conv2_2_w = np.concatenate((o_conv2_2_w, y), axis=1) # (512, 512, 3, 3)

conv3_1_w = original_net.params['conv3_1'][0].data
conv3_1_b = original_net.params['conv3_1'][1].data
o_conv3_1_w = conv3_1_w.copy() # (256, 128, 3, 3)
o_conv3_1_b = conv3_1_b.copy() # (256,)
o_conv3_1_w.resize((512, 128, 3, 3))
o_conv3_1_b.resize((512,))
o_conv3_1_w = np.concatenate((o_conv3_1_w, y), axis=1) # (512, 512, 3, 3)

conv3_2_w = original_net.params['conv3_2'][0].data
conv3_2_b = original_net.params['conv3_2'][1].data
o_conv3_2_w = conv3_2_w.copy() # (256, 256, 3, 3)
o_conv3_2_b = conv3_2_b.copy() # (256,)
o_conv3_2_w.resize((512, 256, 3, 3))
o_conv3_2_b.resize((512,))
z = np.zeros((512, 256, 3, 3), dtype=float)
o_conv3_2_w = np.concatenate((o_conv3_2_w, z), axis=1) # (512, 512, 3, 3)

conv3_3_w = original_net.params['conv3_3'][0].data
conv3_3_b = original_net.params['conv3_3'][1].data
o_conv3_3_w = conv3_3_w.copy() # (256, 256, 3, 3)
o_conv3_3_b = conv3_3_b.copy() # (256,)
o_conv3_3_w.resize((512, 256, 3, 3))
o_conv3_3_b.resize((512,))
o_conv3_3_w = np.concatenate((o_conv3_3_w, z), axis=1) # (512, 512, 3, 3)

conv4_1_w = original_net.params['conv4_1'][0].data
conv4_1_b = original_net.params['conv4_1'][1].data
o_conv4_1_w = conv4_1_w.copy() # (512, 256, 3, 3)
o_conv4_1_b = conv4_1_b.copy() # (512,)
o_conv4_1_w.resize((512, 256, 3, 3))
o_conv4_1_b.resize((512,))
o_conv4_1_w = np.concatenate((o_conv4_1_w, z), axis=1) # (512, 512, 3, 3)

fc6_w = original_net.params['fc6'][0].data
fc6_b = original_net.params['fc6'][1].data
o_fc6_w = fc6_w.copy() # (4096, 25088)
o_fc6_b = fc6_b.copy() # (4096,)
o_fc6_w.resize((6000, 25088))
o_fc6_b.resize((6000,))

fc7_w = original_net.params['fc7'][0].data
fc7_b = original_net.params['fc7'][1].data
o_fc7_w = fc7_w.copy() # (4096, 4096)
o_fc7_b = fc7_b.copy() # (4096,)
w = np.zeros((4096, 1904), dtype=float)
o_fc7_w = np.concatenate((o_fc7_w, w), axis=1) # (4096, 6000)
o_fc7_w.resize((6000, 6000))
o_fc7_b.resize((6000,))

fc8_w = original_net.params['fc8'][0].data
fc8_b = original_net.params['fc8'][1].data
o_fc8_w = fc8_w.copy() # (1000, 4096)
o_fc8_b = fc8_b.copy() # (1000,)
t = np.zeros((1000, 1904), dtype=float)
o_fc8_w = np.concatenate((o_fc8_w, t), axis=1) # (1000, 6000)

# Save obfuscated weights
obfuscated_net = caffe.Net(o_model, 1, weights=weights)

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
obfuscated_net.params['o_conv3_2'][0].data[...] = o_conv3_2_w
obfuscated_net.params['o_conv3_2'][1].data[...] = o_conv3_2_b
obfuscated_net.params['o_conv3_3'][0].data[...] = o_conv3_3_w
obfuscated_net.params['o_conv3_3'][1].data[...] = o_conv3_3_b
obfuscated_net.params['o_conv4_1'][0].data[...] = o_conv4_1_w
obfuscated_net.params['o_conv4_1'][1].data[...] = o_conv4_1_b
obfuscated_net.params['o_fc6'][0].data[...] = o_fc6_w
obfuscated_net.params['o_fc6'][1].data[...] = o_fc6_b
obfuscated_net.params['o_fc7'][0].data[...] = o_fc7_w
obfuscated_net.params['o_fc7'][1].data[...] = o_fc7_b
obfuscated_net.params['o_fc8'][0].data[...] = o_fc8_w
obfuscated_net.params['o_fc8'][1].data[...] = o_fc8_b

obfuscated_net.save('O_VGG_ILSVRC_16_layers_baseline.caffemodel')
