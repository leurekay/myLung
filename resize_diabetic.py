#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""

import os
import PIL
from PIL import Image

import numpy as np


src_dir='/data/coco/val2017'
dis_dir='/data/coco/crop128*128'



def random_crop(src_path,dist_path,width=128,height=128):
    img=Image.open(src_path)
    w,h=img.size
    
    x0=np.random.randint(0,w-width)
    y0=np.random.randint(0,h-height)
    x1=x0+width
    y1=y0+height
    crop=img.crop([x0,y0,x1,y1])
    crop.save(os.path.join(dist_path))
        

def resize(src_path,dist_path,shorter=512):
    img=Image.open(src_path)
    w,h=img.size
    if w<h:
        ww=shorter
        ratio=float(shorter)/w
        hh=h*ratio
    else:
        hh=shorter
        ratio=float(shorter)/h
        ww=w*ratio
    convert=img.resize([int(ww),int(hh)])
    convert.save(os.path.join(dist_path))
    

def resize_crop_square(src_path,dist_path,shorter=299):
    img=Image.open(src_path)
    w,h=img.size
    if w<h:
        ww=shorter
        ratio=float(shorter)/w
        hh=h*ratio
    else:
        hh=shorter
        ratio=float(shorter)/h
        ww=w*ratio
    ww,hh=int(ww),int(hh)
    convert=img.resize([ww,hh])

    start=abs(hh-ww)/2
    end=start+shorter    
    if ww<hh:
        crop=convert.crop([0,start,ww,end])
    else:
        crop=convert.crop([start,0,end,hh])      
    crop.save(os.path.join(dist_path))





if __name__=='__main__':
    if not os.path.exists(dis_dir):
        os.makedirs(dis_dir)
    
    list_names=os.listdir(src_dir)
    
    for count,name in enumerate(list_names):
        src=os.path.join(src_dir,name)
        dist=os.path.join(dis_dir,name)
#        resize_crop_square(src,dist,shorter=128)
        random_crop(src,dist)
        if count%100==0:
            print (count)
    