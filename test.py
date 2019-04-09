# Reference: http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html

import numpy as np
from PIL import Image

import sys
import time
import caffe

model = "models/bvlc_alexnet/deploy.prototxt"
weights = "models/bvlc_alexnet/bvlc_alexnet.caffemodel"

caffe.set_mode_cpu()

#load the model
net = caffe.Net(model, 1, weights=weights)

# load input and configure preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

#note we can change the batch size on-the-fly
#since we classify only one image, we change batch size from 10 to 1
net.blobs['data'].reshape(1,3,227,227)

#load the image in the data layer
im = caffe.io.load_image('examples/images/cat.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data', im)

#compute
start = time.time()
out = net.forward()
end = time.time()
duration = end - start
print ("Wait {0} seconds".format(duration))
#out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))

#predicted predicted class
print out['prob'].argmax()

#print predicted labels
labels = np.loadtxt("data/ilsvrc12/synset_words.txt", str, delimiter='\t')
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
print labels[top_k]
