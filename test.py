# Reference: http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html

import numpy as np
from PIL import Image

import sys
import time
import caffe

model = "models/vgg-16/deploy.prototxt"
weights = "models/vgg-16/VGG_ILSVRC_16_layers.caffemodel"

caffe.set_mode_cpu()

net = caffe.Net(model, 1, weights=weights)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

batch_size = 10
im_cat       = caffe.io.load_image('examples/images/cat.jpg')
im_fish_bike = caffe.io.load_image('examples/images/fish-bike.jpg')

start = time.time()
out = net.forward_all(data=np.asarray([transformer.preprocess('data', im_cat)]));
# out = net.forward_all(data=np.asarray([transformer.preprocess('data', im_cat),
                                       # transformer.preprocess('data', im_cat),
                                       # transformer.preprocess('data', im_cat),
                                       # transformer.preprocess('data', im_cat),
                                       # transformer.preprocess('data', im_cat),
                                       # transformer.preprocess('data', im_fish_bike),
                                       # transformer.preprocess('data', im_fish_bike),
                                       # transformer.preprocess('data', im_fish_bike),
                                       # transformer.preprocess('data', im_fish_bike),
                                       # transformer.preprocess('data', im_fish_bike)]))
end = time.time()
duration = end - start
print ("Inference: {0} seconds".format(duration))

labels = np.loadtxt("data/ilsvrc12/synset_words.txt", str, delimiter='\t')
for i in range(batch_size/10):
    top_k = net.blobs['prob'].data[i].flatten().argsort()[-1:-2:-1]
    print labels[top_k]
