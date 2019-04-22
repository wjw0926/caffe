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

#load the images in the data layer
batch_size = 10
im_cat       = caffe.io.load_image('examples/images/cat.jpg')
im_fish_bike = caffe.io.load_image('examples/images/fish-bike.jpg')

#compute
start = time.time()
out = net.forward_all(data=np.asarray([transformer.preprocess('data', im_cat),
                                       transformer.preprocess('data', im_cat),
                                       transformer.preprocess('data', im_cat),
                                       transformer.preprocess('data', im_cat),
                                       transformer.preprocess('data', im_cat),
                                       transformer.preprocess('data', im_fish_bike),
                                       transformer.preprocess('data', im_fish_bike),
                                       transformer.preprocess('data', im_fish_bike),
                                       transformer.preprocess('data', im_fish_bike),
                                       transformer.preprocess('data', im_fish_bike)]))
end = time.time()
duration = end - start
print ("Wait {0} seconds".format(duration))

#print predicted labels
labels = np.loadtxt("data/ilsvrc12/synset_words.txt", str, delimiter='\t')
for i in range(batch_size):
    top_k = net.blobs['prob'].data[i].flatten().argsort()[-1:-2:-1]
    print labels[top_k]
