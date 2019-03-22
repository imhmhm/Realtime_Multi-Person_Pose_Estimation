#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:50:59 2019

@author: tianlab
"""

import cv2 as cv
import glob
import os

path = '/home/tianlab/hengheng/pose/DukeMTMC-reID/bounding_box_train'
save_path = '/home/tianlab/hengheng/pose/DukeMTMC-reID/pose/train_256'

if not os.path.exists(save_path):
    os.mkdir(save_path)

imlist = sorted(glob.glob(os.path.join(path, '*.jpg')))

for f in imlist:
    filename, ext = os.path.splitext(os.path.basename(f))
    
    im = cv.imread(f)
    im_scale = cv.resize(im, (128, 256), interpolation=cv.INTER_CUBIC)
    cv.imwrite(os.path.join(save_path, filename + '.png'), im_scale)