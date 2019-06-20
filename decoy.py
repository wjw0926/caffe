# Reference: http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html

import numpy as np
from PIL import Image

import sys
import time
import caffe

model = "obfuscator/decoy.prototxt"
weights = "obfuscator/decoy.caffemodel"

caffe.set_mode_cpu()

net = caffe.Net(model, 1, weights=weights)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

im_cat = caffe.io.load_image('examples/images/cat.jpg')

for i in range(200):
    out = net.forward_all(data=np.asarray([transformer.preprocess('data', im_cat)]));
