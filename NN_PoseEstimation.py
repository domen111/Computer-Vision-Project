# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:21:17 2020

@author: User
"""
import json
import cv2
from scipy.spatial.transform import Rotation
import numpy as np
import matplotlib.pyplot as plt
from math import pi

np.random.seed( 1234 )

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
# from keras.optimizers import SGD, Adam
# from keras.utils import np_utils
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

tf.keras.backend.clear_session()

## DNN training parameter
valid_rate = 0.1
epoch_num = 100
batch_size = 32
# target = "MilkJug"
target = "ShampooBottle"
data_path = "DataSet/" + target + "/"
model_pose_path = "model/pose_net_" + target + "(1234)" 

def rotat_mtrx_degree( angle ):
    
    return np.array([[np.cos(np.deg2rad(angle)), - np.sin(np.deg2rad(angle)), 0. ],
                     [np.sin(np.deg2rad(angle)),   np.cos(np.deg2rad(angle)), 0. ],
                     [                       0.,                          0., 1. ]])


def sse(y_true, y_pred):
    """ sum of sqaured errors. """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    
    return K.log( K.sum( K.square( y_pred - y_true ), axis=-1 ) )



def GetCNNModel( img_h, img_w, label_dim, chnnl_num ):
    model = Sequential()
    model.add( Conv2D( 64, kernel_size=(3,3),
                       activation='relu',
                       kernel_initializer='normal',
                       kernel_regularizer=regularizers.l2(1e-4),
                       input_shape=( img_h, img_w, chnnl_num ) ) )
    # model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add( MaxPooling2D( (2,2) ) )
    
    model.add( Conv2D( 64, kernel_size=(3,3),
                       activation='relu',
                       kernel_initializer='normal',
                       kernel_regularizer=regularizers.l2(1e-4) ) )
    # model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add( MaxPooling2D( (2,2) ) )
    
    model.add( Conv2D( 64, kernel_size=(3,3),
                       activation='relu',
                       kernel_initializer='normal',
                       kernel_regularizer=regularizers.l2(1e-4) ) )
    # model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add( MaxPooling2D( (2,2) ) )
    
    model.add( Flatten() )
    
    model.add( Dense( 256,
                      activation='relu',
                      kernel_initializer='normal',
                      kernel_regularizer=regularizers.l2(1e-4)
                    ) )
    model.add( Dropout(0.1) )
    model.add( BatchNormalization() )
    
    model.add( Dense( 256,
                      activation='relu',
                      kernel_initializer='normal',
                      kernel_regularizer=regularizers.l2(1e-4)
                    ) )
    model.add( Dropout(0.1) )
    model.add( BatchNormalization() )
    
    model.add( Dense( label_dim,
                      activation='sigmoid',
                      kernel_initializer='normal',
                      kernel_regularizer=regularizers.l2(1e-4)
                    ) )

    model.compile( loss=sse,
                   optimizer="adam",
                   # metrics=['mse']
                 )
        
    model.summary()
    print("\npredict model built")
    return model


def cut_valid( data_X, data_Y, valid_num ):
    
    X = np.copy( data_X )
    Y = np.copy( data_Y )
    
    # Randomly rearrange the train data
    train_num = len( X )
    
    for i in range( train_num ):
        
        dice = np.random.randint( 0, train_num - 1 )
        X[ [ i, dice ] ] = X[ [ dice, i ] ]
        Y[ [ i, dice ] ] = Y[ [ dice, i ] ]

    X_valid = X[ ( train_num - valid_num ): ]
    X_train = X[ :( train_num - valid_num ) ]
    Y_valid = Y[ ( train_num - valid_num ): ] 
    Y_train = Y[ :( train_num - valid_num ) ]
    
    print('validation data built')
    return X_train , Y_train , X_valid , Y_valid


def PlotTrainingProcess( hist ):
    
    plt.plot( hist.history['loss'] )
    plt.plot( hist.history['val_loss'] )
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    # plt.legend(['train'], loc='upper right')
    plt.show()

    
## train the CNN model
# def Train_CNN_Model( model, X_train, Y_train, X_valid, Y_valid, model_path, to_load_model ):
def Train_CNN_Model( model, X_train, Y_train, X_valid, Y_valid, model_path, to_load_model ):
        
    if( to_load_model == False ):
        
        model.summary()
    
        ## some callbacks
        #    earlystopping = EarlyStopping(monitor='val_acc', patience = 10, verbose=1, mode='max')
        checkpoint = ModelCheckpoint( filepath=( model_path + "_w_best.hdf5" ),
                                      verbose=0,
                                      save_best_only=True,
                                      save_weights_only=True,
                                      monitor='loss',
                                      mode='min'
                                    )
        hist = model.fit( X_train , Y_train,
                           validation_data=( X_valid, Y_valid ), 
                          epochs=epoch_num, batch_size=batch_size,
                          verbose=1,
                          shuffle=True,
        #                      callbacks=[ earlystopping, checkpoint ],
                          callbacks=[ checkpoint ]
                        )

        PlotTrainingProcess( hist )
        model.load_weights( model_path + "_w_best.hdf5" )
        model.save( model_path + '.h5' )
        CNN_model = model
        
    else:
        CNN_model = load_model( model_path + '.h5' )
        
    loss = CNN_model.evaluate( X_valid, Y_valid )
    print("\nValidation loss: " , loss )
    
    print('\nDNN model training and test data label prediction finished.')
    return CNN_model, hist


######################
##### MAIN PART ######
######################

with open( data_path + "scene_gt.json", 'r' ) as read_file:
    data_info = json.load(read_file)

with open( data_path + "scene_gt_info.json", 'r' ) as read_file:
    bbox_info = json.load(read_file)

rgbd_img_set = []
# depth_set = []
i = 0
while(1):
    
    img = cv2.imread( data_path + "rgb/" + str(i).zfill(6) + ".png" )
    depth = cv2.imread( data_path + "depth/" + str(i).zfill(6) + ".png", cv2.IMREAD_UNCHANGED )
    
    if ( hasattr( img, 'dtype' )  ):
        # img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
        depth = depth.reshape(depth.shape[0],depth.shape[1],1)
        img = img.astype('uint16')
        rgbd_img_set.append( np.concatenate( ( img, depth ), axis=2 ) )
        # depth_set.append( depth )
        i = i + 1
    else:
        break

img_num = len(rgbd_img_set)
# orient_set = []
quat_set = []
rotat_set = []
bbox_set = []
for i in range( img_num ):
    
    bbox_set.append( np.array( bbox_info[str(i)][0]['bbox_obj'] ) )
    
    rot_mtrx = np.array( data_info[str(i)][0]['cam_R_m2c'] ).reshape(3,3)
    # orient_set.append( Rotation.from_matrix( rot_mtrx ) )
    rotat_set.append( rot_mtrx )
    # quat = Rotation.from_matrix( rot_mtrx ).as_quat()
    # quat_set.append( quat )
    # theta = np.arccos( quat[0] ) * 2 # [rad]
    # s = np.sin( theta / 2 )
    # quat_set.append( np.array([theta * 180./pi, s * quat[1], s * quat[2], s * quat[3] ]) )

bbox_set = np.vstack( bbox_set )
img_w = int( np.linalg.norm( [ np.max(bbox_set[:,2]), np.max(bbox_set[:,3]) ], ord=2 ) )
img_h = img_w

## tune image size
for i in range( img_num ):
    
    rgbd_img_set[i] = rgbd_img_set[i][ bbox_set[i,1] - int(( img_h - bbox_set[i,3] ) / 2):
                                     bbox_set[i,1] + int(( img_h + bbox_set[i,3] ) / 2),
                                     bbox_set[i,0] - int(( img_w - bbox_set[i,2] ) / 2):
                                     bbox_set[i,0] + int(( img_w + bbox_set[i,2] ) / 2),
                                      :
                                     ]
    if( rgbd_img_set[i].shape[0] >= img_h ): rgbd_img_set[i] = rgbd_img_set[i][:img_h-1,:,:]
    if( rgbd_img_set[i].shape[1] >= img_w ): rgbd_img_set[i] = rgbd_img_set[i][:,:img_w-1,:]

img_w = img_w - 1
img_h = img_h - 1

## augment data set
augmented_factor = 3
angle_set = np.arange( 1, augmented_factor ) * 360 / augmented_factor
for rotat_degree in angle_set:
    M = cv2.getRotationMatrix2D( (img_w/2,img_h/2), - rotat_degree, 1 )
    for j in range(img_num):
        rgbd_img_set.append( cv2.warpAffine( rgbd_img_set[j], M, (img_w, img_h)) )
        rotat_set.append( rotat_mtrx_degree( rotat_degree ).dot( rotat_set[j] ) )

img_num = len( rgbd_img_set )

for i in range( img_num ):
    quat = Rotation.from_matrix( rotat_set[i] ).as_quat()
    quat_set.append( quat )

quat_set = np.vstack( quat_set )
label_dim = quat_set.shape[1]

shrink_rate = 2
for i in range(img_num):
    rgbd_img_set[i] = cv2.resize( rgbd_img_set[i], ( int(img_w/shrink_rate), int(img_h/shrink_rate) ), interpolation=cv2.INTER_AREA )
img_w = int(img_w/shrink_rate)
img_h = int(img_h/shrink_rate)

rgbd_img_set = np.array( rgbd_img_set )
# rgbd_img_set = rgbd_img_set.reshape( img_num, img_h, img_w, 4 )
# rgb_img_set = rgb_img_set.reshape( img_num, img_h, img_w, 1 )

## Data normalization to 0 ~ 1
rgbd_img_set = rgbd_img_set.astype('float')
rgbd_img_set[:,:,:,0:3] = rgbd_img_set[:,:,:,0:3] / 255.
rgbd_img_set[:,:,:,3] = ( rgbd_img_set[:,:,:,3] - np.min( rgbd_img_set[:,:,:,3] ) ) / ( np.max( rgbd_img_set[:,:,:,3] ) - np.min( rgbd_img_set[:,:,:,3] ) )
quat_set = ( quat_set + 1.0 ) / 2.0

X_train, Y_train, X_valid, Y_valid = cut_valid( rgbd_img_set, quat_set, int( 0.1 * img_num ) )


CNN_model_pose = GetCNNModel( img_h, img_w, label_dim, rgbd_img_set.shape[3] )
CNN_model_pose, hist_pose = Train_CNN_Model( CNN_model_pose, X_train, Y_train, X_valid, Y_valid, model_pose_path, to_load_model=False )
np.savetxt( model_pose_path + "_loss_pose(1234).csv", hist_pose.history['loss'] )



