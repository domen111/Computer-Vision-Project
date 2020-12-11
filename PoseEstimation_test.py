# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:21:17 2020

@author: User
"""
import json
import cv2
from scipy.spatial.transform import Rotation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import tensorflow as tf
import winsound

tf.keras.backend.clear_session()
from PoseEstimation_library import rot_mtrx_deg, custom_loss, GetCNNModel, \
cut_valid, PlotTrainingProcess, Train_CNN_Model
import keras.losses
keras.losses.custom_loss = custom_loss

np.random.seed( 1234 )

tf.keras.backend.clear_session()

## DNN training parameter
valid_rate = 0.1
epoch_num = 250
batch_size = 32
# target = "MilkJug"
# target = "ShampooBottle"
target = "Cube"
data_path = "DataSet/" + target + "/"
model_pose_path = "model/pose_net_v5" + target + "(1234)" 
loadModel = True

######################
##### MAIN PART ######
######################

cntr_set_l = np.loadtxt( "cntr_pt_set1_l.csv", delimiter="," ).astype('int')
cntr_set_r = np.loadtxt( "cntr_pt_set1_r.csv", delimiter="," ).astype('int')

img_num = len(cntr_set_l)


rgb_img_set_raw_l = []
# rgb_img_set_raw_r= []
for i in range( 1, img_num+1 ):
    
    img_L = cv2.imread( data_path + "Video1/l/img" + str(i).zfill(4) + "l.png" )
    img_L = cv2.cvtColor( img_L, cv2.COLOR_BGR2RGB )
    # img_R = cv2.imread( data_path + "Video1/r/img" + str(i).zfill(4) + "r.png" )
    # img_R = cv2.cvtColor( img_R, cv2.COLOR_BGR2RGB )
    depth = cv2.imread( data_path + "Video1/disparity/img" + str(i).zfill(4) + "disparity.png", cv2.IMREAD_UNCHANGED )
    depth = depth.reshape(depth.shape[0],depth.shape[1],1)
    
    img_L = img_L.astype('uint16')   
    rgb_img_set_raw_l.append( np.concatenate( ( img_L, depth ), axis=2 ) )
    # rgb_img_set_raw_l.append( img_L )
    # rgb_img_set_raw_r.append( img_R )

img_w = 330
img_h = img_w
rgb_img_set_l = []
# rgb_img_set_r = []
## tune image size
for i in range( img_num ):
    
    rgb_img_set_l.append( np.copy( rgb_img_set_raw_l[i][ int(cntr_set_l[i,0] - img_h/2) :
                                                         int(cntr_set_l[i,0] + img_h/2) ,
                                                         int(cntr_set_l[i,1] - img_w/2) :
                                                         int(cntr_set_l[i,1] + img_w/2) ,
                                                         :
                                                       ] ) )
    # rgb_img_set_r.append( np.copy( rgb_img_set_raw_r[i][ int(cntr_set_r[i,0] - img_h/2) :
    #                                                      int(cntr_set_r[i,0] + img_h/2) ,
    #                                                      int(cntr_set_r[i,1] - img_w/2) :
    #                                                      int(cntr_set_r[i,1] + img_w/2) ,
    #                                                      :
    #                                                    ] ) )


# for i in range(10):
#     idx = np.random.randint(len(rgb_img_set))
#     idx = 139
#     # y_test = quat_set[idx] * 2 - 1
#     print(idx)
#     rot_test = rotat_set[idx]
#     plt.imshow( rgb_img_set[idx] )
#     plt.arrow( img_w/2, img_w/2 ,100*(rot_test[1,0]),-100*(rot_test[2,0]), width=4, color='b' )
#     plt.arrow( img_w/2, img_w/2 ,100*(rot_test[1,1]),-100*(rot_test[2,1]), width=4, color='b' )
#     plt.arrow( img_w/2, img_w/2 ,100*(rot_test[1,2]),-100*(rot_test[2,2]), width=4, color='b' )   
#     plt.show()



shrink_rate = 3
for i in range(img_num):
    rgb_img_set_l[i] = cv2.resize( rgb_img_set_l[i], ( int(img_w/shrink_rate), int(img_h/shrink_rate) ), interpolation=cv2.INTER_AREA )
    # rgb_img_set_r[i] = cv2.resize( rgb_img_set_r[i], ( int(img_w/shrink_rate), int(img_h/shrink_rate) ), interpolation=cv2.INTER_AREA )

img_w = int(img_w/shrink_rate)
img_h = int(img_h/shrink_rate)

rgb_img_set = []
# for i in range(img_num):
#     rgb_img_set.append( np.concatenate( ( rgb_img_set_l[i], rgb_img_set_r[i] ), axis=2 ) )
    
# rgb_img_set = np.array( rgb_img_set )
# rgb_img_set = rgb_img_set[:,:,:,0:3]

# rgb_img_set = np.array( rgb_img_set )
# rgb_img_set = rgb_img_set[:,:,:,0:3]

rgb_img_set = np.array( rgb_img_set_l )
del rgb_img_set_l
# del rgb_img_set_l, rgb_img_set_r

channel_num = rgb_img_set.shape[3]
# rgb_img_set = rgb_img_set.reshape( img_num, img_h, img_w, 4 )
# rgb_img_set = rgb_img_set.reshape( img_num, img_h, img_w, 1 )

## Data normalization to 0 ~ 1
rgb_img_set = rgb_img_set.astype('float')
rgb_img_set[:,:,:,0:3] = rgb_img_set[:,:,:,0:3] / 255.
# rgb_img_set[:,:,:,0:6] = rgb_img_set[:,:,:,0:6] / 255.
rgb_img_set[:,:,:,3] = ( rgb_img_set[:,:,:,3] - np.min( rgb_img_set[:,:,:,3] ) ) / ( np.max( rgb_img_set[:,:,:,3] ) - np.min( rgb_img_set[:,:,:,3] ) )

tf.keras.backend.clear_session()
CNN_model_pose, hist_pose = Train_CNN_Model( None, None, None, None, None, model_pose_path, \
                                             0, 0, to_load_model=loadModel )

if( hist_pose!= None ):
    np.savetxt( model_pose_path + "_loss_pose.csv", hist_pose.history['loss'] )


for i in range(img_num):
    # idx = np.random.randint(len(rgb_img_set))
    y_prdct = CNN_model_pose.predict(rgb_img_set[i,:,:,:].reshape(1,img_h,img_w,channel_num)).reshape(-1)
    y_prdct = y_prdct * 2 - 1
    # y_prdct = y_prdct / np.linalg.norm( y_prdct )
    rot_prdct = Rotation.from_quat( y_prdct ).as_matrix()
    plt.imshow( rgb_img_set_raw_l[i][:,:,0:3] )
    plt.axis('off')
    plt.arrow( cntr_set_l[i,1], cntr_set_l[i,0], 100*(rot_prdct[1,0]),-100*(rot_prdct[2,0]), width=15, color='r', label='X' )
    plt.arrow( cntr_set_l[i,1], cntr_set_l[i,0], 100*(rot_prdct[1,1]),-100*(rot_prdct[2,1]), width=15, color='g', label='Y' )
    plt.arrow( cntr_set_l[i,1], cntr_set_l[i,0], 100*(rot_prdct[1,2]),-100*(rot_prdct[2,2]), width=15, color='b', label='Z' )
    plt.savefig( "Results/Video1_LandD/"+ str(i).zfill(4) + "l.png" )
    plt.show()

'''
for i in range(10):
    idx = np.random.randint(len(_img_set))
    y_test = quat_set[idx] * 2 - 1
    rot_test = Rotation.from_quat( y_test ).as_matrix()
    plt.imshow( _img_set[idx] )
    plt.arrow( img_w/2, img_w/2 ,50*(rot_test[0,0]),50*(rot_test[2,0]), width=2, color='b' )
    plt.arrow( img_w/2, img_w/2 ,50*(rot_test[0,1]),50*(rot_test[2,1]), width=2, color='b' )
    plt.arrow( img_w/2, img_w/2 ,50*(rot_test[0,2]),50*(rot_test[2,2]), width=2, color='b' )   
    plt.show()
'''   

winsound.Beep( 400, 500 )
