#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:42:04 2018

@author: ly
"""

import SimpleITK as sitk
import numpy as np
import pandas as pd
import csv
import os,time
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing

def readCSV(filename):
    lines = []
    with open(filename, "rb") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines


subset0_dir='/data/lungCT/luna/subset0'
annotations_path='/data/lungCT/luna/annotations.csv'
candidate_path='/data/lungCT/luna/candidates.csv'
candidate2_path='/data/lungCT/luna/pull_aiserver/candidates_V2.csv'
submit_path='/data/lungCT/luna/sampleSubmission.csv'

qualified_path='/home/ly/lung/dsb2017_grt123/training/detector/labels/lunaqualified.csv'
shorter_path='/home/ly/lung/dsb2017_grt123/training/detector/labels/shorter.csv'
annos_path='/home/ly/lung/dsb2017_grt123/training/detector/labels/annos.csv'
label_qualified_path='/home/ly/lung/dsb2017_grt123/training/detector/labels/label_qualified.csv'
label_job0_full_path='/home/ly/lung/dsb2017_grt123/training/detector/labels/label_job0_full.csv'
label_job1_path='/home/ly/lung/dsb2017_grt123/training/detector/labels/label_job1.csv'

file_list=os.listdir(subset0_dir)
id_list=filter(lambda x:x.split('.')[-1]=='mhd' ,file_list)
df_a=pd.read_csv(annotations_path)
df_c=pd.read_csv(candidate_path)
df_c2=pd.read_csv(candidate2_path)
df_s=pd.read_csv(submit_path)
df_q=pd.read_csv(qualified_path)
df_shorter=pd.read_csv(shorter_path)

df_annos=pd.read_csv(annos_path)
df_label_qualified=pd.read_csv(label_qualified_path)
df_label_job0_full=pd.read_csv(label_job0_full_path)
df_label_job1=pd.read_csv(label_job1_path)

group_c2=df_c2.groupby('seriesuid').count()
group_anno=df_annos.groupby('seriesuid').count()

patients_id=list(group_c2.index)
#for i in range(len(patients_id)):
#    id0=patients_id[i]
#    id0_xyz=df_c[(df_c['seriesuid']==id0) & (df_c['class']==1)]
#    coords=id0_xyz[['coordX','coordY','coordZ']]
#    coords=np.array(coords)
#    
#    
#    
#    fig = plt.figure() 
#    ax = fig.add_subplot(111, projection = '3d')
#    ax.scatter(coords[:,0], coords[:,1], coords[:,2], c = 'c', marker = 'o')
#    plt.title(str(i))
#
#    if not os.path.exists('image'):
#        os.makedirs('image')
#    plt.savefig(os.path.join('image',str(i)+'.jpg'))

 
    
    


#patient0=file_list[0]
#patient0_mhd=os.path.join(subset0_dir,patient0)
#itkimage=sitk.ReadImage(patient0_mhd)
#numpyImage=sitk.GetArrayFromImage(itkimage)

#a=load_itk_image(patient0_mhd)