#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:20:17 2018

@author: ly
"""

import inputdata
import data
from model_3D import *
import layers

import os
from PIL import Image
import tensorflow as tf


import keras
import keras.backend as K
from keras.models import Model,load_model


import numpy as np
import pandas as pd


config = {}
config['anchors'] = [ 5.0, 10.0, 20.]
config['chanel'] = 1
config['crop_size'] = [128, 128, 128]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 6. #mm
config['sizelim2'] = 30
config['sizelim3'] = 40
config['aug_scale'] = True
config['r_rand_crop'] = 0.3
config['pad_value'] = 170
config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38','990fbe3f0a1b53878669967b9afd1441','adc3bbc63d40f8761c59be10f1e504c3']
config['train_over_total']=0.9





data_dir='/data/lungCT/luna/temp/luna_npy'
SAVED_MODEL='/data/lungCT/luna/temp/model/my_model.h5'

        
        
dataset=data.DataBowl3Detector(data_dir,config)
patch,label,coord=dataset.__getitem__(295)
a=label[:,:,:,:,0]

#def iter_data(data_dir):
#     dataset=data.DataBowl3Detector(data_dir,config)
#     for i in range(10000):
#         yield dataset.__getitem__(i)
# 
#data_iterator=iter_data(data_dir)
# 
#box=[]
#for count,i in  enumerate(data_iterator):
#    box.append(i[0][0][50][50][50])
#    print (count)



#loss function
myloss=layers.myloss


#load model
if os.path.exists(SAVED_MODEL):
    print ("*************************\n restore model\n*************************")
    model=load_model(SAVED_MODEL)  
else:
    model=n_net()
    
adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam,
              loss=myloss,
              )



# numbers of sample correspoding train and val
train_dataset=data.DataBowl3Detector(data_dir,config,phase='train')
train_samples=train_dataset.__len__()
val_dataset=data.DataBowl3Detector(data_dir,config,phase='val')
val_samples=val_dataset.__len__()



class LossHistory(keras.callbacks.Callback):
    def on_epoch_begin(self,epoch, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        print ('epoch:',epoch,'    ',self.losses)
history = LossHistory()

def generate_arrays(phase):
    dataset=data.DataBowl3Detector(data_dir,config,phase=phase)
    n_samples=dataset.__len__()
    while True:
        for i in range(n_samples):
            x, y ,_ = dataset.__getitem__(i)
            x=np.expand_dims(x,axis=-1)
            y=np.expand_dims(y,axis=0)
            yield (x, y)

model.fit_generator(generate_arrays(phase='train'),
                    steps_per_epoch=train_samples,epochs=3,
                    verbose=1,callbacks=[history],
                    validation_data=generate_arrays('val'),
                    validation_steps=val_samples,
                    workers=4,)
model_dir=SAVED_MODEL.split(SAVED_MODEL.split('/')[-1])[0]
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save(SAVED_MODEL,include_optimizer=False)