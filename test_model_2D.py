#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:58:04 2018

@author: ly
"""

import os
from PIL import Image


import numpy as np
import pandas as pd


from keras.models import Model
from keras.layers import Dense, Dropout, Flatten,Input,Activation
from keras.layers import Conv2D, MaxPooling2D,Conv3D,Deconv2D
from keras.layers.merge import concatenate
from keras.optimizers import SGD

from keras.utils import np_utils  
from keras.utils import plot_model  


import resnet

def res_block(x,conv_filters,pool_size,pool_strides):
    for i in range(3):  
        x=resnet.basic_block2(filters=conv_filters)(x)
        x=Activation(activation='relu')(x)
    x=MaxPooling2D(pool_size=pool_size,strides=pool_strides)(x)    
    return x


image_dir='/data/coco/crop128*128_'
box=[]
file_list=os.listdir(image_dir)
for  i,name in enumerate(file_list):
    if i ==210:
        print name
    path=os.path.join(image_dir,name)
    img=Image.open(path)
    box.append(np.array(img))
X=np.array(box)


input_img = Input(shape=(96,96,3))

#first 2 conv layers
x = Conv2D(24, (3, 3), padding='same', activation='relu')(input_img)
x = Conv2D(24, (3, 3), strides=(1,1),padding='same', activation='relu')(x)


#first residual block,including 3 residual units
"""
resnet block get from github
https://github.com/leurekay/keras-resnet/blob/master/resnet.py
"""
x=res_block(x,conv_filters=32,pool_size=(2,2),pool_strides=(2,2))

#next 3 residual block
r2=res_block(x,conv_filters=64,pool_size=(2,2),pool_strides=(2,2))
r3=res_block(r2,conv_filters=64,pool_size=(2,2),pool_strides=(2,2))
r4=res_block(r3,conv_filters=64,pool_size=(2,2),pool_strides=(2,2))

#feedback path
x=Deconv2D(64,kernel_size=(2,2),strides=2)(r4)
x=concatenate([r3,x])
x=res_block(x,64,(1,1),(1,1))
x=Deconv2D(64,kernel_size=(2,2),strides=2)(x)
x=concatenate([r2,x])
x=res_block(x,128,(1,1),(1,1))

#predictions = Dense(10, activation='softmax')(x)
model=Model(inputs=input_img,outputs=x )
model.summary()

plot_model(model, to_file='images/model.png',show_shapes=True)
