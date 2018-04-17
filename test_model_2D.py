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
from keras.layers import Conv2D, MaxPooling2D,Conv3D
from keras.optimizers import SGD

from keras.utils import np_utils  
from keras.utils import plot_model  


import resnet





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


input_img = Input(shape=(128,128,3))

#first 2 conv layers
x = Conv2D(24, (3, 3), padding='same', activation='relu')(input_img)
x = Conv2D(24, (3, 3), strides=(1,1),padding='same', activation='relu')(x)


#first resnet block
"""
resnet block get from github
https://github.com/leurekay/keras-resnet/blob/master/resnet.py
"""
for i in range(3):  
    x=resnet.basic_block2(filters=32)(x)
    x=Activation(activation='relu')(x)
x=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

#next 3 resnet blocks
for j in range(3):
    for i in range(3):  
        x=resnet.basic_block2(filters=64)(x)
        x=Activation(activation='relu')(x)
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)    


#predictions = Dense(10, activation='softmax')(x)
model=Model(inputs=input_img,outputs=x )
model.summary()

plot_model(model, to_file='model.png',show_shapes=False)
