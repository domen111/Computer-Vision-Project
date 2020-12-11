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
from matplotlib.font_manager import FontProperties
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
epoch_num = 200
batch_size = 32
# target = "MilkJug"
# target = "ShampooBottle"
target = "Cube"
data_path = "DataSet/" + target + "/"
model_pose_path = "model/pose_net_v7" + target + "(1234)" 
loadModel = True

######################
##### MAIN PART ######
######################
data_info = pd.read_csv( data_path + "training_data_gt.csv")
bbox_set_l = np.loadtxt( "bbox_set.csv", delimiter="," ).astype('int')
bbox_set_r = np.loadtxt( "bbox_set_r.csv", delimiter="," ).astype('int')
data_info_item = data_info.columns.tolist()
data_info_value = data_info.values.astype('float')
img_num = len(data_info)

rotat_set = []
rgb_img_set_l = []
rgb_img_set_r= []
for i in range( img_num ):
    rotat_set.append( rot_mtrx_deg( data_info_value[i,2], 'Z' ).dot( rot_mtrx_deg( data_info_value[i,1], 'Y' ) ) )
    img_L = cv2.imread( data_path + "PostProcessing/" + str(i).zfill(4) + "l_dirt.png" )
    img_L = cv2.cvtColor( img_L, cv2.COLOR_BGR2RGB )
    img_R = cv2.imread( data_path + "PostProcessing/" + str(i).zfill(4) + "r_dirt.png" )
    img_R = cv2.cvtColor( img_R, cv2.COLOR_BGR2RGB )
    # depth = cv2.imread( data_path + "PostProcessing/" + str(i).zfill(4) + "d.png", cv2.IMREAD_UNCHANGED )
    # depth = depth.reshape(depth.shape[0],depth.shape[1],1)
    # img_L = img_L.astype('uint16')
    
    # rgb_img_set.append( np.concatenate( ( img_L, img_R ), axis=2 ) )
    rgb_img_set_l.append( img_L )
    rgb_img_set_r.append( img_R )

img_w = int( np.linalg.norm( [ np.max(bbox_set_l[:,2]), np.max(bbox_set_l[:,3]) ], ord=2 ) )
img_h = int( np.linalg.norm( [ np.max(bbox_set_r[:,2]), np.max(bbox_set_r[:,3]) ], ord=2 ) )
img_w = np.max( [ img_w, img_h ] )
img_h = img_w

## tune image size
for i in range( img_num ):
    
    rgb_img_set_l[i] = rgb_img_set_l[i][ max( 0, bbox_set_l[i,1] - int(( img_h - bbox_set_l[i,3] ) / 2) ):
                                         bbox_set_l[i,1] + int(( img_h + bbox_set_l[i,3] ) / 2),
                                         max( 0, bbox_set_l[i,0] - int(( img_w - bbox_set_l[i,2] ) / 2) ):
                                         bbox_set_l[i,0] + int(( img_w + bbox_set_l[i,2] ) / 2),
                                          :
                                       ]
    rgb_img_set_r[i] = rgb_img_set_r[i][ max( 0, bbox_set_r[i,1] - int(( img_h - bbox_set_r[i,3] ) / 2) ):
                                          bbox_set_r[i,1] + int(( img_h + bbox_set_r[i,3] ) / 2),
                                          max( 0, bbox_set_r[i,0] - int(( img_w - bbox_set_r[i,2] ) / 2) ):
                                          bbox_set_r[i,0] + int(( img_w + bbox_set_r[i,2] ) / 2),
                                          :
                                        ]
    if( rgb_img_set_l[i].shape[0] >= img_h ): rgb_img_set_l[i] = rgb_img_set_l[i][:img_h-1,:,:]
    if( rgb_img_set_l[i].shape[1] >= img_w ): rgb_img_set_l[i] = rgb_img_set_l[i][:,:img_w-1,:]
    if( rgb_img_set_r[i].shape[0] >= img_h ): rgb_img_set_r[i] = rgb_img_set_r[i][:img_h-1,:,:]
    if( rgb_img_set_r[i].shape[1] >= img_w ): rgb_img_set_r[i] = rgb_img_set_r[i][:,:img_w-1,:]

img_w = img_w - 1
img_h = img_h - 1


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


## augment data set
augmented_factor = 18
angle_set = np.arange( 1, augmented_factor ) * 360 / augmented_factor
for rotat_degree in angle_set:
    M = cv2.getRotationMatrix2D( (img_w/2,img_h/2), rotat_degree, 1 )
    for j in range(img_num):
        rgb_img_set_l.append( cv2.warpAffine( rgb_img_set_l[j], M, (img_w, img_h)) )
        rgb_img_set_r.append( cv2.warpAffine( rgb_img_set_r[j], M, (img_w, img_h)) )
        rotat_set.append( rot_mtrx_deg( rotat_degree, 'X' ).dot( rotat_set[j] ) )

img_num = len( rgb_img_set_l )

quat_set = []
for i in range( img_num ):
    quat = Rotation.from_matrix( rotat_set[i] ).as_quat() # (x, y, z, w)
    quat_set.append( quat )

quat_set = np.vstack( quat_set )
label_dim = quat_set.shape[1]

shrink_rate = 3
for i in range(img_num):
    rgb_img_set_l[i] = cv2.resize( rgb_img_set_l[i], ( int(img_w/shrink_rate), int(img_h/shrink_rate) ), interpolation=cv2.INTER_AREA )
    rgb_img_set_r[i] = cv2.resize( rgb_img_set_r[i], ( int(img_w/shrink_rate), int(img_h/shrink_rate) ), interpolation=cv2.INTER_AREA )

img_w = int(img_w/shrink_rate)
img_h = int(img_h/shrink_rate)

rgb_img_set = []
for i in range(img_num):
    rgb_img_set.append( np.concatenate( ( rgb_img_set_l[i], rgb_img_set_r[i] ), axis=2 ) )
    
# rgb_img_set = np.array( rgb_img_set )
rgb_img_set = np.array( rgb_img_set_l )
del rgb_img_set_l, rgb_img_set_r


channel_num = rgb_img_set.shape[3]
# rgb_img_set = rgb_img_set.reshape( img_num, img_h, img_w, 4 )
# rgb_img_set = rgb_img_set.reshape( img_num, img_h, img_w, 1 )

## Data normalization to 0 ~ 1
rgb_img_set = rgb_img_set.astype('float')
# rgb_img_set[:,:,:,0:3] = rgb_img_set[:,:,:,0:3] / 255.
rgb_img_set[:,:,:,0:6] = rgb_img_set[:,:,:,0:6] / 255.
# rgb_img_set[:,:,:,3] = ( rgb_img_set[:,:,:,3] - np.min( rgb_img_set[:,:,:,3] ) ) / ( np.max( rgb_img_set[:,:,:,3] ) - np.min( rgb_img_set[:,:,:,3] ) )
quat_set = ( quat_set + 1.0 ) / 2.0

X_train, Y_train, X_valid, Y_valid = cut_valid( rgb_img_set, quat_set, int( 0.1 * img_num ) )


CNN_model_pose = GetCNNModel( img_h, img_w, label_dim, rgb_img_set.shape[3] )
# CNN_model_pose = None
CNN_model_pose, hist_pose = Train_CNN_Model( CNN_model_pose, X_train, Y_train, X_valid, Y_valid, model_pose_path, \
                                             batch_size, epoch_num, to_load_model=loadModel )
if( hist_pose!= None ):
    np.savetxt( model_pose_path + "_loss_pose.csv", hist_pose.history['loss'] )


for i in range(10):
    fontP = FontProperties()
    fontP.set_size('medium')
    idx = np.random.randint(len(X_valid))
    y_test = Y_valid[idx] * 2 - 1
    y_prdct = CNN_model_pose.predict(X_valid[idx,:,:,:].reshape(1,img_h,img_w,channel_num)).reshape(-1)
    y_prdct = y_prdct * 2 - 1
    # y_prdct = y_prdct / np.linalg.norm( y_prdct )
    print( "y_test:", y_test )
    print( "y_predict:", y_prdct )
    rot_prdct = Rotation.from_quat( y_prdct ).as_matrix()
    rot_test = Rotation.from_quat( y_test ).as_matrix()
    plt.imshow( X_valid[idx,:,:,0:3] )
    plt.axis('off')
    a1 = plt.arrow( img_w/2, img_w/2 ,50*(rot_prdct[1,0]),-50*(rot_prdct[2,0]), width=2, color='r', label='X_prdct' )
    a2 = plt.arrow( img_w/2, img_w/2 ,50*(rot_prdct[1,1]),-50*(rot_prdct[2,1]), width=2, color='g', label='Y_prdct' )
    a3 = plt.arrow( img_w/2, img_w/2 ,50*(rot_prdct[1,2]),-50*(rot_prdct[2,2]), width=2, color='b', label='Z_prdct' )
    a4 = plt.arrow( img_w/2, img_w/2 ,50*(rot_test[1,0]),-50*(rot_test[2,0]), width=2, color='violet', label='X_test' )
    a5 = plt.arrow( img_w/2, img_w/2 ,50*(rot_test[1,1]),-50*(rot_test[2,1]), width=2, color='palegreen', label='Y_test' )
    a6 = plt.arrow( img_w/2, img_w/2 ,50*(rot_test[1,2]),-50*(rot_test[2,2]), width=2, color='aqua', label='Z_test' )   
    plt.legend(handles=[ a1, a2, a3, a4, a5, a6 ], bbox_to_anchor=(1.35, 1), loc='upper right', prop=fontP )
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
