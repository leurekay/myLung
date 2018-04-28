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
    
    y_neg_true= y_neg_true[:30,:]
    
    
    y_pos_pred=tf.boolean_mask(y_pred,mask_pos)
    y_neg_pred=tf.boolean_mask(y_pred,mask_neg)
    y_neg_pred=y_neg_pred[:30,:]
    
    y_true=tf.concat((y_pos_true,y_neg_true),axis=0)
    y_pred=tf.concat((y_pos_pred,y_neg_pred),axis=0)
    loss_cls=tf.losses.log_loss(y_true[:,0],y_pred[:,0])
    loss_reg=tf.reduce_sum(tf.square(y_true[:,1:5]-y_pred[:,1:5]),axis=1)
    loss_reg=tf.multiply(y_true[:,0],loss_reg)
    loss_reg=tf.reduce_mean(loss_reg)
    loss_cls=tf.cast(loss_cls,tf.float32)
    loss=tf.add(loss_cls,loss_reg)
    return loss


def myloss2():
    pass

model.compile(optimizer='adam',
              loss=myloss,)


model.fit(X,y,
          batch_size=1,epochs=1,)