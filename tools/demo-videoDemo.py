#!/usr/bin/env python
 
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
from newnms.nms import  soft_nms
from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
import h5py
'''CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
'''
CLASSES = ('__background__','human')
#NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',),'res152':('res152_faster_rcnn_iter_1190000.ckpt',)}
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',),'res152':('res152.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),'coco':('coco_2014_train+coco_2014_valminusminival',)}

def vis_detections(im, image_name, class_name, dets,xminarr,yminarr,xmaxarr,ymaxarr,results,score_file,index_file,num_boxes, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return num_boxes

    im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        # ax.add_patch(
        #     plt.Rectangle((bbox[0], bbox[1]),
        #                   bbox[2] - bbox[0],
        #                   bbox[3] - bbox[1], fill=False,
        #                   edgecolor='red', linewidth=3.5)
        #     )
        num_boxes = num_boxes+1
        results.write("{}\n".format(image_name))
        score_file.write("{}\n".format(score))
        xminarr.append(int(round(bbox[0])));yminarr.append(int(round(bbox[1])));xmaxarr.append(int(round(bbox[2])));ymaxarr.append(int(round(bbox[3])))
        # ax.text(bbox[0], bbox[1] - 2,
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')

    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #               fontsize=14)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.draw()
    return num_boxes

def demo(sess, net, idx,xminarr,yminarr,xmaxarr,ymaxarr,results,score_file,index_file,num_boxes,imagedir):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'coco-minival_images', image_name)
    im_file = os.path.join(imagedir, str(idx)+'.jpg')
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)

    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.7
    # Visualize people
    cls_ind = 1 
    cls = 'person'
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]

    #Prev image
    im_file = os.path.join(imagedir, str(idx-1)+'.jpg')
    if os.path.isfile(im_file):
        im = cv2.imread(im_file)
        scores, boxes = im_detect(sess, net, im)

        # Visualize people
        cls_ind = 1 
        cls_boxes = np.append(cls_boxes,boxes[:, 4*cls_ind:4*(cls_ind + 1)], axis=0)
        cls_scores = np.append(cls_scores,scores[:, cls_ind], axis=0)

    #Prev Prev image
    im_file = os.path.join(imagedir, str(idx-2)+'.jpg')
    if os.path.isfile(im_file):
        im = cv2.imread(im_file)
        scores, boxes = im_detect(sess, net, im)

        # Visualize people
        cls_ind = 1 
        cls_boxes = np.append(cls_boxes,boxes[:, 4*cls_ind:4*(cls_ind + 1)], axis=0)
        cls_scores = np.append(cls_scores,scores[:, cls_ind], axis=0)

    #Next image
    im_file = os.path.join(imagedir, str(idx+1)+'.jpg')
    if os.path.isfile(im_file):
        im = cv2.imread(im_file)
        scores, boxes = im_detect(sess, net, im)

        # Visualize people
        cls_ind = 1 
        cls_boxes = np.append(cls_boxes,boxes[:, 4*cls_ind:4*(cls_ind + 1)], axis=0)
        cls_scores = np.append(cls_scores,scores[:, cls_ind], axis=0)

    #Next Next image
    im_file = os.path.join(imagedir, str(idx+2)+'.jpg')
    if os.path.isfile(im_file):
        im = cv2.imread(im_file)
        scores, boxes = im_detect(sess, net, im)

        # Visualize people
        cls_ind = 1 
        cls_boxes = np.append(cls_boxes,boxes[:, 4*cls_ind:4*(cls_ind + 1)], axis=0)
        cls_scores = np.append(cls_scores,scores[:, cls_ind], axis=0)

    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    #keep = nms(dets, NMS_THRESH)
    keep=soft_nms(dets,method=2)
    
    #dets = dets[keep, :]
    dets=keep
    if(dets.shape[0]!=0):
        index_file.write("{} {} ".format(str(idx)+'.jpg',num_boxes+1))
    num_boxes = vis_detections(im, str(idx)+'.jpg', cls, dets,xminarr,yminarr,xmaxarr,ymaxarr,results,score_file,index_file,num_boxes, thresh=CONF_THRESH)
    if(dets.shape[0]!=0):
        index_file.write("{}\n".format(num_boxes))
    # for cls_ind, cls in enumerate(CLASSES[1:]):
    #     cls_ind += 1 # because we skipped background
    #     cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    #     cls_scores = scores[:, cls_ind]
    #     dets = np.hstack((cls_boxes,
    #                       cls_scores[:, np.newaxis])).astype(np.float32)
    #     keep = nms(dets, NMS_THRESH)
    #     dets = dets[keep, :]
    #     vis_detections(im, cls, dets, thresh=CONF_THRESH)
    return num_boxes




def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res152')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='coco')
    parser.add_argument('--inputpath',dest='inputpath',help='image-directory')
    parser.add_argument('--outputpath',dest='outputpath',help='output-directory')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('../output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])
    inputpath=args.inputpath
    outputpath=args.outputpath
    #print(tfmodel)	
    #import time
    #time.sleep(10)
 
    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    elif demonet =='res152':
        net = resnetv1(num_layers=152)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 81,
                          tag='default', anchor_scales=[2,4,8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    videoLen = 1022 
    #im_names = [line.rstrip('\n') .rstrip('\r') for line in open("../data/test-dev-list.txt")]
    #im_names = ['3.jpg','300.jpg','500.jpg','200.jpg','400.jpg','712.jpg']
    xminarr=[]
    yminarr=[]
    xmaxarr=[]
    ymaxarr=[]
    results = open("videoDemo-"+outputpath+"/test-dev_images.txt", 'w')
    score_file = open("videoDemo-"+outputpath+"/score-proposals.txt",'w')
    index_file = open("videoDemo-"+outputpath+"/index.txt",'w')
    #results = open("test-mpii-softnms/test-dev_images.txt", 'w')
    #score_file = open("test-mpii-softnms/score-proposals.txt",'w')
    #index_file = open("test-mpii-softnms/index.txt",'w')   
    num_boxes = 0
    for vid in range(videoLen):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for {}'.format(vid+1))
        num_boxes=demo(sess, net, vid+1,xminarr,yminarr,xmaxarr,ymaxarr,results,score_file,index_file,num_boxes,inputpath)
    with h5py.File("videoDemo-"+outputpath+"/test-dev.h5", 'w') as hf:
                    hf.create_dataset('xmin', data=np.array(xminarr))
                    hf.create_dataset('ymin', data=np.array(yminarr))
                    hf.create_dataset('xmax', data=np.array(xmaxarr))
                    hf.create_dataset('ymax', data=np.array(ymaxarr))
    results.close()    
    score_file.close()
    index_file.close()
