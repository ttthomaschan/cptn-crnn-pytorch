#-*- coding:utf-8 -*-
#'''
# Created on 18-12-11 上午10:09
#
#'''
import os

# base_dir = 'path to dataset base dir'
base_dir = './images'
img_dir = os.path.join(base_dir, 'VOC2007_text_detection/JPEGImages')
xml_dir = os.path.join(base_dir, 'VOC2007_text_detection/Annotations')

icdar19_mlt_img_dir = '/home/elimen/Data/OCR_dataset/MLT2019/Task1/Images/'
#icdar19_mlt_img_dir = '/home/tthom/storage/DATA/MLT2019/Images/'
#icdar19_mlt_img_dir = '/mnt/data/Data/MLT2019/Images/'
icdar19_mlt_gt_dir = '/home/elimen/Data/OCR_dataset/MLT2019/Task1/train_gt_t13/'
#icdar19_mlt_gt_dir = '/home/tthom/storage/DATA/MLT2019/train_gt_t13/'
#icdar19_mlt_gt_dir = '/mnt/data/Data/MLT2019//train_gt_t13/'

num_workers = 8
pretrained_weights = 'checkpoints/ctpn_epoch50_0.3801_0.0971_0.4773.pth'

 

anchor_scale = 16
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300

# bgr can find from  here: https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py
IMAGE_MEAN = [123.68, 116.779, 103.939]
OHEM = True

checkpoints_dir = './checkpoints'
outputs = r'./logs'
