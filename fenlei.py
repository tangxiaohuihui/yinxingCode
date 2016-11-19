#coding=utf-8
#加载必要的库
import numpy as np
import string
import sys,os
import imghdr
import caffe

# #设置当前目录
# caffe_root = '/home/abc/caffe/' 
# sys.path.insert(0, caffe_root + 'python')
# test_images0 = 0
# test_images1 = 0
# test_images2 = 0
# right_classify0 = 0
# right_classify1 = 0
# right_classify2 = 0
# import caffe
# os.chdir(caffe_root)

net_file="D:/caffe/caffe-master/examples/newbag/deploy.prototxt"
caffe_model="D:/caffe/caffe-master/data/myself/train1/result_iter_1000.caffemodel"
mean_file='D:/caffe/caffe-master/examples/newbag/mean.npy'
imagenet_labels_filename = "D:/caffe/caffe-master/data/myself/train1/test_words.txt"
image = "D:/caffe/caffe-master/data/myself/train1/test/149-1.jpg"

# fp = open(biaoqian_file)
# images = fp.readlines()

net = caffe.Net(net_file,caffe_model,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255) 
transformer.set_channel_swap('data', (2,1,0))


            
im=caffe.io.load_image(image)
net.blobs['data'].data[...] = transformer.preprocess('data',im)
out = net.forward()


#imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
#for i in np.arange(top_k.size):
#print top_k[0], labels[top_k[0]]
true_label = image[3]
print true_label,top_k[0]

