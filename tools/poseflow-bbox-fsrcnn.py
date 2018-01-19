# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import h5py
import scipy.io as sio

import sys
import random
import math
import skimage.io

import utils
from utils.timer import Timer
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
from newnms.nms import  soft_nms
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
from tqdm import tqdm

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__','human')
#NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',),'res152':('res152_faster_rcnn_iter_1190000.ckpt',)}
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',),'res152':('res152.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),'coco':('coco_2014_train+coco_2014_valminusminival',)}


# image_dir = "/home/yuliang/data/MultiPerson_PoseTrack_v0.1/videos_align"
# image_dir = '/home/yuliang/data/mpii-video-pose/'
image_dir = '/home/yuliang/data/posetrack_data/posetrack_data'
# list_file = '/home/yuliang/code/PoseFlow/listfiles/bonn-small'
# list_file = '/home/yuliang/code/PoseFlow/listfiles/mpii-video-pose'
list_file = '/home/yuliang/code/PoseFlow/listfiles/bonn-big'

test_list_file = os.path.join(list_file, 'test_list.txt')
# test_list_file = os.path.join(list_file, 'test_list_align.txt')
# test_list_file = os.path.join(list_file, 'all_list.txt')


lines = [line.rstrip('\n') .rstrip('\r') for line in open(test_list_file)]

# directory = '/home/yuliang/code/PoseFlow/dataset/generated_bbox/fsrcnn-arg/bonn-small/'
directory = '/home/yuliang/code/PoseFlow/dataset/generated_bbox/fsrcnn-arg/bonn-big/test/'

# directory = '/home/yuliang/code/PoseFlow/dataset/generated_bbox/mask-rcnn/mpii-video-pose/'
# directory = '/home/yuliang/code/PoseFlow/dataset/generated_bbox/mask-rcnn/bonn-big/'

if not os.path.exists(directory):
    os.makedirs(directory)
results = open(directory+"test-bbox_images.txt", 'w')
score_file = open(directory+"score.txt",'w')
index_file = open(directory+"index.txt",'w')

FileLength = len(lines)


num_boxes=0

xminarr=[]
yminarr=[]
xmaxarr=[]
ymaxarr=[]


# In[ ]:


cfg.TEST.HAS_RPN = True  # Use RPN for proposals

# model path
tfmodel = os.path.join('../output', 'res152', DATASETS['coco'][0], 'default',
                          NETS['res152'][0])
if not os.path.isfile(tfmodel + '.meta'):
    raise IOError(('{:s} not found.\nDid you download the proper networks from '
                   'our server and place them properly?').format(tfmodel + '.meta'))

# set config
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True

# init session
sess = tf.Session(config=tfconfig)
net = resnetv1(num_layers=152)
net.create_architecture("TEST", 81,
                      tag='default', anchor_scales=[2,4,8, 16, 32])
saver = tf.train.Saver()
saver.restore(sess, tfmodel)
print('Loaded network {:s}'.format(tfmodel))

def detect(sess,net,im):
    scores, boxes = im_detect(sess, net, im)
    cls_boxes = boxes[:, [5,4,7,6]]
    cls_scores = scores[:, 1]
    return cls_boxes, cls_scores


# In[ ]:


def nozero_mean(matrix):
    return np.sum(matrix)/(np.sum(matrix!=0.0)+1)

def op_argument(det, flow, img_height, img_width):
    #ymin xmin ymax xmax
    dets = det.copy()
    flow[flow<0.5] = 0.0   
    for row,det in enumerate(dets):
        ymin,xmin,ymax,xmax,score = det
        deltaX = nozero_mean(flow[int(ymin):int(ymax),int(xmin):int(xmax),0])
        deltaY = nozero_mean(flow[int(ymin):int(ymax),int(xmin):int(xmax),1])
        det = [np.clip(ymin+deltaY,0,img_height), np.clip(xmin+deltaX,0,img_width), np.clip(ymax+deltaY,0,img_height), np.clip(xmax+deltaX,0,img_width), score*0.95]
        dets[row] = det
    return dets

for i in tqdm(range(FileLength)):

    if lines[i].split("\t")[0].split(".")[0][-4:] == "crop":
        continue
    
    [vid_name, img_name_mid] = lines[i].split("\t")[0][:-12], lines[i].split("\t")[0][-12:]
    # [vid_name, img_name_mid] = [lines[i].split("\t")[0].split('/')[0],lines[i].split("\t")[0].split('/')[1]]

    img_id_mid = int(img_name_mid.split(".")[0])
    img_ids = [img_id_mid-2, img_id_mid-1, img_id_mid+1, img_id_mid+2]
    img_names = ["%s/%08d.jpg"%(vid_name,img_id) for img_id in img_ids]
    
    filename_mid = os.path.join(image_dir, vid_name, img_name_mid)
    image_mid = skimage.io.imread(filename_mid)
    boxes, scores = detect(sess,net,image_mid)
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
    
    for img_name in img_names:
        
        filename = os.path.join(image_dir, img_name)
        if os.path.exists(filename):
            image = skimage.io.imread(filename)
            [img_height, img_width,_] = image.shape
            boxes, scores = detect(sess,net,image)
            dets_ = np.hstack((boxes,scores[:, np.newaxis])).astype(np.float32)
            # prvs = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # next = cv2.cvtColor(image_mid,cv2.COLOR_BGR2GRAY)
            # opflow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # dets = np.vstack((dets, op_argument(dets_, opflow, img_height, img_width)))
            dets = np.vstack((dets,dets_))
    
    keep=soft_nms(dets,method=2)
    detections = keep
    # print(detections.shape)
        
#     Parse the outputs.
    det_conf = detections[:,4]
    det_xmin = detections[:,1]
    det_ymin = detections[:,0]
    det_xmax = detections[:,3]
    det_ymax = detections[:,2]

    top_indices1 = [m for m, conf in enumerate(det_conf) if conf > 0.1]
    top_indices3 = [m for m, height in enumerate(det_ymax-det_ymin) if height > 0.1*img_height]
    
    top_indices = list(set(top_indices1) & set(top_indices3))
#     print(len(top_indices))
    
    top_conf = det_conf[top_indices]
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    if(top_conf.shape[0]!=0):
        index_file.write("{} {} ".format(os.path.join(vid_name,img_name_mid),num_boxes+1))
    for k in range(top_conf.shape[0]):
        
        xmin = int(round(top_xmin[k]))
        ymin = int(round(top_ymin[k]))
        xmax = int(round(top_xmax[k]))
        ymax = int(round(top_ymax[k]))
        score = top_conf[k]
        
#         print(image.shape[0], image.shape[1])
#         print(xmin, xmax, ymin, ymax, score)
        if xmin>=xmax or ymin>=ymax or xmax>image.shape[1] or ymax>image.shape[0]:
            print('error '+img_name_mid)
            
        xminarr.append(xmin);
        yminarr.append(ymin);
        xmaxarr.append(xmax);
        ymaxarr.append(ymax);
        
        results.write("{}\n".format(os.path.join(vid_name,img_name_mid)))
        score_file.write("{}\n".format(score))

        num_boxes += 1
    
    if(top_conf.shape[0]!=0):
        index_file.write("{}\n".format(num_boxes))


# In[ ]:


print("Average Boxes per image:", float(num_boxes)/FileLength)
results.close()    
score_file.close()
index_file.close()
with h5py.File(directory+'test-bbox.h5', 'w') as hf:
                hf.create_dataset('xmin', data=np.array(xminarr))
                hf.create_dataset('ymin', data=np.array(yminarr))
                hf.create_dataset('xmax', data=np.array(xmaxarr))
                hf.create_dataset('ymax', data=np.array(ymaxarr))
print("Done")

print(num_boxes)
