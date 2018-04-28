#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:20:47 2018

@author: ly
"""

import numpy as np
import tensorflow as tf


x=tf.constant([[1,0.2,0.5,0.2,5.3],[0,0,0,0,0],[-10,-10,-10,-10,-10]])
x_=tf.constant([[0.8,0.6,0.6,1.9,5],[0.1,0,0,0,0],[-10,-10,-10,-10,-10]])

key=tf.constant([0.,0,0])

g=tf.greater(x[:,0],key)
#w=tf.where(g,x,y)

index=tf.boolean_mask(x,g)

conca=tf.gather(x,[1,2])

loss=tf.reduce_sum(tf.square(x-x_),axis=1)
loss_mean=tf.reduce_mean(loss)






groundtruth_save_path='/data/lungCT/luna/temp/patch/groundtruthes.npy'
y_data=np.load(groundtruth_save_path)[10:18]

y_true=tf.constant(y_data,tf.float32)
y_true=tf.reshape(y_true,[-1,5])
y_pred=np.random.random([8,32,32,32,3,5])
y_pred=tf.constant(y_pred,tf.float32)
y_pred=tf.reshape(y_pred,[-1,5])

#
#
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
#loss=loss_cls+


init=tf.global_variables_initializer()
sess=tf.Session()
#
yy=sess.run(y_pos_pred)
yy_=sess.run(y_neg_pred)
#
m=sess.run(mask_neg)
yy_true=sess.run(y_true)
yy_pred=sess.run(y_pred)

loss=sess.run(loss_reg)




def myloss(y_true, y_pred):
    mask_pos=tf.greater(y_true[:,0],0.5)
    mask_neg=tf.equal(y_true[:,0],tf.constant([0],dtype='float64'))
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
    loss_cls=tf.cast(loss_cls,tf.float64)
    loss=tf.add(loss_cls,loss_reg)   
    return loss
#
#gg=sess.run(g)
#ggg=sess.run(conca)
#gggg=sess.run(index)