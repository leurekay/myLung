#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:43:47 2018

@author: ly
"""

import os
from PIL import Image


import numpy as np
import pandas as pd


from keras.models import Model
from keras.layers import Dense, Dropout, Flatten,Input,Activation,Reshape
from keras.layers import Conv2D, MaxPooling2D,MaxPooling3D,Conv3D,Deconv2D,Deconv3D
from keras.layers.merge import concatenate
from keras.optimizers import SGD

from keras.utils import np_utils  
from keras.utils import plot_model  

import resnet3D





def res_block(x,conv_filters,pool_size,pool_strides):
    for i in range(3):  
        x=resnet3D.basic_block2(filters=conv_filters)(x)
        x=Activation(activation='relu')(x)
    x=MaxPooling3D(pool_size=pool_size,strides=pool_strides)(x)    
    return x





def n_net():
    

    input_img = Input(shape=(128,128,128,1))
    
    #first 2 conv layers
    x = Conv3D(24, (3, 3,3), padding='same', activation='relu')(input_img)
    x = Conv3D(24, (3, 3,3), strides=(1,1,1),padding='same', activation='relu')(x)
    
    
    #first residual block,including 3 residual units
    """
    resnet block get from github
    https://github.com/leurekay/keras-resnet/blob/master/resnet.py
    """
    x=res_block(x,conv_filters=32,pool_size=(2,2,2),pool_strides=(2,2,2))
    
    #next 3 residual block
    r2=res_block(x,conv_filters=64,pool_size=(2,2,2),pool_strides=(2,2,2))
    r3=res_block(r2,conv_filters=64,pool_size=(2,2,2),pool_strides=(2,2,2))
    r4=res_block(r3,conv_filters=64,pool_size=(2,2,2),pool_strides=(2,2,2))
    
    
    #feedback path
    x=Deconv3D(64,kernel_size=(2,2,2),strides=2)(r4)
    x=concatenate([r3,x])
    x=res_block(x,64,(1,1,1),(1,1,1))
    x=Deconv3D(64,kernel_size=(2,2,2),strides=2)(x)
    x=concatenate([r2,x])
    x=res_block(x,128,(1,1,1),(1,1,1))
    x= Conv3D(64, (3, 3,3), strides=(1,1,1),padding='same', activation='relu')(x)
    x= Conv3D(15, (3, 3,3), strides=(1,1,1),padding='same', activation='sigmoid')(x)
    x= Reshape((32,32,32,3,5))(x)
    
    #predictions = Dense(10, activation='softmax')(x)
    model=Model(inputs=input_img,outputs=x )
    return model

def n_net2():
    

    input_img = Input(shape=(128,128,128,1,))
    
    #first 2 conv layers
    x = Conv3D(24, (3, 3,3), padding='same', activation='relu')(input_img)
    x = Conv3D(24, (3, 3,3), strides=(1,1,1),padding='same', activation='relu')(x)
    x = Conv3D(15 ,(4, 4,4), strides=(4,4,4),padding='same', activation='relu')(x)
    
    
    

    x= Reshape((32,32,32,3,5))(x)
    
    #predictions = Dense(10, activation='softmax')(x)
    model=Model(inputs=input_img,outputs=x )
    return model


if __name__=='__main__':
    model=n_net2()

    model.summary()
    
    plot_model(model, to_file='images/model3d.png',show_shapes=True)