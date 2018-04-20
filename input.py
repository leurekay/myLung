#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:28:14 2018

@author: ly
"""

import numpy as np
import os

def random_cropCube(tensor,target_shape):
    nx,ny,nz=tensor.shape
    mx,my,mz=target_shape
    x0=np.random.randint(0,nx-mx)
    y0=np.random.randint(0,ny-my)
    z0=np.random.randint(0,nz-mz)
    x1=x0+mx
    y1=y0+my
    z1=z0+mz
    return tensor[x0:x1,y0:y1,z0:z1]


dsb_dir='/data/lungCT/dsb2017/sample_processing'
luna_dir='/data/lungCT/dsb2017/generation_data'


file_list=os.listdir(dsb_dir)
file_list=filter(lambda x:x.split('.')[-1]=='npy',file_list)
ct_list=filter(lambda x:x.split('_')[-1]=='clean.npy',file_list)
cube_box=[]
for series_id in ct_list:
    path=os.path.join(dsb_dir,series_id)
    cube_box.append(np.load(path)[0])



file_list2=os.listdir(luna_dir)
file_list2=filter(lambda x:x.split('.')[-1]=='npy',file_list2)
ct_list2=filter(lambda x:x.split('_')[-1]=='clean.npy',file_list2)
cube_box2=[]
for series_id in ct_list2:
    path=os.path.join(luna_dir,series_id)
    cube_box2.append(np.load(path)[0])

a=random_cropCube(cube_box2[0],[128,128,128])  

#npy_matrix=np.load(os.path.join(dsb_dir,ct_list[0])) 