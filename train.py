#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 15:54:11 2018

@author: ly
"""

import inputdata
from model_3D import *

import os
from PIL import Image
import tensorflow as tf


import keras
import keras.backend as K


import numpy as np
import pandas as pd

patch_save_path='/data/lungCT/luna/temp/patch/patches.npy'
groundtruth_save_path='/data/lungCT/luna/temp/patch/groundtruthes.npy'


#X,y=inputdata.generate_feeddata()
X=np.load(patch_save_path)
X=X.astype('float32')
y=np.load(groundtruth_save_path)

model=n_net()



def myloss(y_true, y_pred):
    mask_pos=tf.greater(y_true[:,0],0.5)
    mask_neg=tf.equal(y_true[:,0],tf.constant([0],dtype='float32'))
    y_pos_true=tf.boolean_mask(y_true,mask_pos)
    y_neg_true=tf.boolean_mask(y_true,mask_neg)
    
    y_pos_pred=tf.boolean_mask(y_pred,mask_pos)
    y_neg_pred=tf.boolean_mask(y_pred,mask_neg)
    
#    box_true=[]
#    box_pred=[]
#    for i in range(2):
#        rand=np.random.randint(0,32*2)
#        slice_true=y_neg_true[rand]
#        box_true.append(tf.reshape(slice_true,[1,5]))
#        slice_pred=y_neg_pred[rand]
#        box_pred.append(tf.reshape(slice_pred,[1,5]))
#    y_neg_true=tf.concat(box_true,axis=0)
#    y_neg_pred=tf.concat(box_pred,axis=0)
    
    y_neg_true=y_neg_true[:30]
    y_neg_pred=y_neg_pred[:30]
    
    y_true=tf.concat((y_pos_true,y_neg_true),axis=0)
    y_pred=tf.concat((y_pos_pred,y_neg_pred),axis=0)
    
    loss_cls=tf.losses.log_loss(y_true[:,0],y_pred[:,0])
    #loss_cls2=tf.reduce_mean(y_true[:,0]*tf.log(y_pred[:,0])+(1-y_true[:,0])*tf.log(1-y_pred[:,0]))
    
    def smoothL1(x,y):
        """
        x,y :both are tensors with same shape
        """
        mask=tf.greater(tf.abs(x-y),1)
        l=tf.where(mask,tf.abs(x-y),tf.square(x-y))
        return tf.reduce_sum(l,axis=1)
        
    loss_reg=y_true[:,0]*smoothL1(y_true[:,1:5],y_pred[:,1:5]) 
    loss_reg=tf.reduce_mean(loss_reg)   
    
    #loss_cls=tf.cast(loss_cls,tf.float32)
    loss=tf.add(loss_cls,loss_reg)
    return loss


def myloss2():
    pass


adam=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(optimizer=adam,
              loss=myloss,)


model.fit(X,y,
          batch_size=1,epochs=10,)