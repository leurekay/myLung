#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:08:02 2018

@author: ly
"""
import os

import numpy as np
import pandas as pd

from PIL import Image


def data_label(image_dir,label_path):
    image_name_list=os.listdir(image_dir)
    df_label=pd.read_csv(label_path)
    n_imgs=len(image_name_list)
    n_labels=df_label.shape[0]
    
    if n_imgs<n_labels:
        #only use the sub-data, let label.csv correspond to images
        image_name_list2=[x.split('.')[0] for x in image_name_list]
    #    image_name_list2=map(lambda x:x.split('.')[0],image_name_list)
        df_label['flag']=df_label.apply(lambda x:x['image'] in image_name_list2,axis=1)
        df_label=df_label[df_label['flag']==True]
        df_label.drop(labels=['flag'],axis=1,inplace=True)
    
    label_index_img_list=df_label["image"].tolist()  
    y=np.array(df_label['level'])
    box=[]
    box2=[]
    for i,name in enumerate(label_index_img_list):
        if i%4!=0:
            continue
        a=Image.open(os.path.join(image_dir,name)+'.jpeg')
        box.append(np.array(a))  
        box2.append(y[i])
        
    return [np.array(box),np.array(box2)]

if __name__=='__main__':
    IMAGE_DIR='../data/square_'
    LABEL_PATH='../data/trainLabels.csv'
    X,y=data_label(IMAGE_DIR,LABEL_PATH)