#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:28:14 2018

@author: ly
"""

import numpy as np
import os




def cropPatch_by_random(img3d,crop_size):
    nx,ny,nz=img3d.shape
    mx,my,mz=crop_size
    x0=np.random.randint(0,nx-mx)
    y0=np.random.randint(0,ny-my)
    z0=np.random.randint(0,nz-mz)
    x1=x0+mx
    y1=y0+my
    z1=z0+mz
    return img3d[x0:x1,y0:y1,z0:z1]
    

def cropPatch_by_nodule(img3d,target,crop_size=[128,128,128],bound_size=12):
    """
    img3d:3D numpy
    target:[x,y,z,diameter] numpy or list
    crop_size:  e.g.[128,128,128]
    bound_size: boundary margin
    """
    crop_size=np.array(crop_size)
    start = []
    for i in range(3):
        r = target[3] / 2
        s = np.floor(target[i] - r)+ 1 - bound_size
        e = np.ceil (target[i] + r)+ 1 + bound_size - crop_size[i] 
        if s>e:
            start.append(np.random.randint(e,s))#!
        else:
            start.append(int(target[i])-crop_size[i]/2+np.random.randint(-bound_size/2,bound_size/2))        
    
    start=np.array(start)
    end=start+crop_size
    start=[max(0,start[i]) for i in range(len(start))]

    
    cube=img3d[start[0]:end[0],start[1]:end[1],start[2]:end[2]]
    #if size<128,padding boundary with 170
    cube=np.pad(cube, ((0,crop_size[0]-cube.shape[0]),(0,crop_size[1]-cube.shape[1]),(0,crop_size[2]-cube.shape[2])), 'constant', constant_values=170)
    
    start.append(0)
    start=np.array(start)
    new_target=np.round(target-start).astype('int')    
    return cube,new_target


def IoU(box0, box1):
    """
    box0: numpy [x,y,z,diameter]
    """
    
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / float(union)

def groundtruth_of_anchor(anno):
    map_size=[32,32,32]
    block_size=4
    rpn_scale=[5,10,20]
    threshold_up=0.4
    threshold_low=0.02
    interval_point=np.array([2+i*4 for i in range(32)])

    box=np.zeros(map_size+[3,5])
    for i in range(32):
        for j in range(32):
            for k in range(32):
                for s in range(3):
                    Ax,Ay,Az,Ar=interval_point[i],interval_point[j],interval_point[k],rpn_scale[s]
                    anchor=np.array([Ax,Ay,Az,Ar])
                    iou=IoU(anchor,anno)
                    if iou>threshold_up:
#                        print (iou)
                        dx=(anno[0]-Ax)/float(Ar)
                        dy=(anno[1]-Ay)/float(Ar)
                        dz=(anno[2]-Az)/float(Ar)
                        dr=np.log(anno[3]/float(Ar))
                        box[i,j,k,s,:]=[1,dx,dy,dz,dr]
                    elif iou<threshold_low:
                        box[i,j,k,s,:]=[0,0,0,0,0]
                    else:#    def __getitem__(self, idx,split=None):
#        t = time.time()
#        np.random.seed(int(str(t%1)[2:7]))#seed according to time
#
#        isRandomImg  = False
#        if self.phase !='test':
#            if idx>=len(self.bboxes):
#                isRandom = True
#                idx = idx%len(self.bboxes)
#                isRandomImg = np.random.randint(2)
#            else:
#                isRandom = False
#        else:
#            isRandom = False
#        
#        if self.phase != 'test':
#            if not isRandomImg:
#                bbox = self.bboxes[idx]
#                filename = self.filenames[int(bbox[0])]
#                imgs = np.load(filename)
#                bboxes = self.sample_bboxes[int(bbox[0])]
#                isScale = self.augtype['scale'] and (self.phase=='train')
#                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes,isScale,isRandom)
#                if self.phase=='train' and not isRandom:
#                     sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
#                        ifflip = self.augtype['flip'], ifrotate=self.augtype['rotate'], ifswap = self.augtype['swap'])
#            else:
#                randimid = np.random.randint(len(self.kagglenames))
#                filename = self.kagglenames[randimid]
#                imgs = np.load(filename)
#                bboxes = self.sample_bboxes[randimid]
#                isScale = self.augtype['scale'] and (self.phase=='train')
#                sample, target, bboxes, coord = self.crop(imgs, [], bboxes,isScale=False,isRand=True)
#            label = self.label_mapping(sample.shape[1:], target, bboxes)
#            sample = (sample.astype(np.float32)-128)/128
#            #if filename in self.kagglenames and self.phase=='train':
#            #    label[label==-1]=0
#            return torch.from_numpy(sample), torch.from_numpy(label), coord
#        else:
#            imgs = np.load(self.filenames[idx])
#            bboxes = self.sample_bboxes[idx]
#            nz, nh, nw = imgs.shape[1:]
#            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
#            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
#            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
#            imgs = np.pad(imgs, [[0,0],[0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',constant_values = self.pad_value)
#            
#            xx,yy,zz = np.meshgrid(np.linspace(-0.5,0.5,imgs.shape[1]/self.stride),
#                                   np.linspace(-0.5,0.5,imgs.shape[2]/self.stride),
#                                   np.linspace(-0.5,0.5,imgs.shape[3]/self.stride),indexing ='ij')
#            coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32')
#            imgs, nzhw = self.split_comber.split(imgs)
#            coord2, nzhw2 = self.split_comber.split(coord,
#                                                   side_len = self.split_comber.side_len/self.stride,
#                                                   max_stride = self.split_comber.max_stride/self.stride,
#                                                   margin = self.split_comber.margin/self.stride)
#            assert np.all(nzhw==nzhw2)
#            imgs = (imgs.astype(np.float32)-128)/128
#            return torch.from_numpy(imgs), bboxes, torch.from_numpy(coord2), np.array(nzhw)
#
#    def __len__(self):
#        if self.phase == 'train':
#            return len(self.bboxes)/(1-self.r_rand)
#        elif self.phase =='val':
#            return len(self.bboxes)
#        else:
#            return len(self.sample_bboxes)
                        box[i,j,k,s,:]=[-10,-10,-10,-10,-10]
    return box
    
 

def generate_feeddata():
#    luna_dir='/data/lungCT/luna/temp/luna_small'
    luna_dir='/data/lungCT/luna/temp/luna_npy/small'
    
    patch_save_dir='/data/lungCT/luna/temp/patch'
    if not os.path.exists(patch_save_dir):
        os.makedirs(patch_save_dir)

    patient_list=os.listdir(luna_dir)
    
    ct_list=filter(lambda x:x.split('_')[-1]=='clean.npy',patient_list)
    label_list=filter(lambda x:x.split('_')[-1]=='label.npy',patient_list)
    
    id_list_by_ct=map(lambda x:x.split('_')[0],ct_list)
    id_list_by_label=map(lambda x:x.split('_')[0],label_list)

    id_list=set.intersection(set(id_list_by_ct),set(id_list_by_label))
    id_list=list(id_list)    
    
    img3d_box=[]
    label_box=[]
    
    patch_box=[]
    groundtruth_box=[]
    for i,index in enumerate(id_list):
        if i%10==0:
            print (i)
        path_ct=os.path.join(luna_dir,index+'_clean.npy')
        path_label=os.path.join(luna_dir,index+'_label.npy')
        img3d=np.load(path_ct)[0]
        label=np.load(path_label)
        if int(sum(sum(label)))==0:#indicate no nodule
            patch=cropPatch_by_random(img3d,[128,128,128])
            groundtruth=np.zeros([32,32,32,3,5],'int16')
#            print (index)
        else:
            patch,anno=cropPatch_by_nodule(img3d,label[np.random.randint(len(label))])
            groundtruth=groundtruth_of_anchor(anno)
        patch_box.append(patch)
        groundtruth_box.append(groundtruth)
        
    patch_box=np.array(patch_box)
    patch_box=np.expand_dims(patch_box,axis=-1)
    groundtruth_box=np.array(groundtruth_box) 
    np.save(os.path.join(patch_save_dir,'patches.npy'),patch_box)
    np.save(os.path.join(patch_save_dir,'groundtruthes.npy'),groundtruth_box)
    return patch_box,groundtruth_box


if __name__=='__main__':
    X,y=generate_feeddata()
    
    
 
#    luna_dir='/data/lungCT/luna/temp/luna_small'
#    
#    patch_save_dir='/data/lungCT/luna/temp/patch'
#    if not os.path.exists(patch_save_dir):
#        os.makedirs(patch_save_dir)
#
#    patient_list=os.listdir(luna_dir)
#    
#    ct_list=filter(lambda x:x.split('_')[-1]=='clean.npy',patient_list)
#    label_list=filter(lambda x:x.split('_')[-1]=='label.npy',patient_list)
#    
#    id_list_by_ct=map(lambda x:x.split('_')[0],ct_list)
#    id_list_by_label=map(lambda x:x.split('_')[0],label_list)
#
#    id_list=set.intersection(set(id_list_by_ct),set(id_list_by_label))
#    id_list=list(id_list)    
#    
#    img3d_box=[]
#    label_box=[]
#    
#    patch_box=[]
#    groundtruth_box=[]
#    for index in id_list:
#        path_ct=os.path.join(luna_dir,index+'_clean.npy')
#        path_label=os.path.join(luna_dir,index+'_label.npy')
#        img3d=np.load(path_ct)[0]
#        label=np.load(path_label)
#        patch_box.append(img3d)
#        label_box.append(label)
    
            

