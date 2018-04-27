#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 15:54:11 2018

@author: ly
"""

import inputdata
from model_3D import n_net

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
y=np.load(groundtruth_save_path)
model=n_net()


def myloss(y_true, y_pred):
    y_true=K.reshape(y_true,[-1,5])
    y_pred=K.reshape(y_pred,[-1,5])
    return  tf.reduce_mean( K.mean(y_true-y_pred, axis=-1))

model.compile(optimizer='adam',
              loss=myloss,)

model.fit(X,y,
          batch_size=1,epochs=1,)