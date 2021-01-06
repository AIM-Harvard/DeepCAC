"""
  ----------------------------------------
    Heart segmentation - DL model arch.
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 2.7.17
  ----------------------------------------
  
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

from tensorflow.python.keras.utils import *
from tensorflow.python.keras.engine import Input
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, Dropout


def dice_coef(y_true, y_pred, smooth=1.):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

## ----------------------------------------
## ----------------------------------------

def dice_coef_loss(y_true, y_pred):
  return -dice_coef(y_true, y_pred)

## ----------------------------------------
## ----------------------------------------

def getUnet3d(down_steps, input_shape, pool_size=(2, 2, 2), conv_size=(3, 3, 3), initial_learning_rate=0.00001,
              mgpu=1, ext=False, drop_out=0.5):
  if down_steps == 4:
    if not ext:
      return getUnet3d_4(input_shape, pool_size, conv_size, initial_learning_rate, mgpu)
    else:
      return getUnet3d_4_ext(input_shape, pool_size, conv_size, initial_learning_rate, mgpu, drop_out)
  else:
    print 'Wrong U-Net parameters specified ("down_steps")'

## ----------------------------------------
## ----------------------------------------

def getUnet3d_4(input_shape, pool_size, conv_size, initial_learning_rate, mgpu):
  inputs = Input(input_shape, name='model_input')
  conv1 = Conv3D(32, conv_size, activation='relu', padding='same', name='conv_1_1')(inputs)
  conv1 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_1_2')(conv1)
  pool1 = MaxPooling3D(pool_size=pool_size, name='pool_1')(conv1)

  conv2 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_2_1')(pool1)
  conv2 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_2_2')(conv2)
  pool2 = MaxPooling3D(pool_size=pool_size, name='pool_2')(conv2)

  conv3 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_3_1')(pool2)
  conv3 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_3_2')(conv3)
  pool3 = MaxPooling3D(pool_size=pool_size, name='pool_3')(conv3)

  conv4 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_4_1')(pool3)
  conv4 = Conv3D(512, conv_size, activation='relu', padding='same', name='conv_4_2')(conv4)
  pool4 = MaxPooling3D(pool_size=pool_size, name='pool_4')(conv4)

  conv5 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_5_1')(pool4)
  conv5 = Conv3D(512, conv_size, activation='relu', padding='same', name='conv_5_2')(conv5)

  up6 = UpSampling3D(size=pool_size, name='up_6')(conv5)
  up6 = concatenate([up6, conv4], axis=4, name='conc_6')
  conv6 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_6_1')(up6)
  conv6 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_6_2')(conv6)

  up7 = UpSampling3D(size=pool_size, name='up_7')(conv6)
  up7 = concatenate([up7, conv3], axis=4, name='conc_7')
  conv7 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_7_1')(up7)
  conv7 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_7_2')(conv7)

  up8 = UpSampling3D(size=pool_size, name='up_8')(conv7)
  up8 = concatenate([up8, conv2], axis=4, name='conc_8')
  conv8 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_8_1')(up8)
  conv8 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_8_2')(conv8)

  up9 = UpSampling3D(size=pool_size, name='up_9')(conv8)
  up9 = concatenate([up9, conv1], axis=4, name='conc_9')
  conv9 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_9_1')(up9)
  conv9 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_9_2')(conv9)

  conv10 = Conv3D(1, (1, 1, 1), name='conv_10')(conv9)
  act = Activation('sigmoid', name='act')(conv10)

  if mgpu == 1:
    print 'Compiling single GPU model'
    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss,
                  metrics=[dice_coef])
    return model
  elif mgpu > 1:
    print 'Compiling multi GPU model'
    model = Model(inputs=inputs, outputs=act)
    parallel_model = multi_gpu_model(model, gpus=mgpu)
    parallel_model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss,
                           metrics=[dice_coef])
    return parallel_model
  else:
    print 'ERROR Wrong number of GPUs defined'
    return

## ----------------------------------------
## ----------------------------------------

def getUnet3d_4_ext(input_shape, pool_size, conv_size, initial_learning_rate, mgpu, drop_out):
  print 'Use extended MGPU 4 model'
  inputs = Input(input_shape, name='model_input')
  conv1 = Conv3D(32, conv_size, activation='relu', padding='same', name='conv_1_1')(inputs)
  norm1 = BatchNormalization(axis=4, name='norm_1_1')(conv1)
  conv1 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_1_2')(norm1)
  norm1 = BatchNormalization(axis=4, name='norm_1_2')(conv1)
  pool1 = MaxPooling3D(pool_size=pool_size, name='pool_1')(norm1)

  conv2 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_2_1')(pool1)
  norm2 = BatchNormalization(axis=4, name='norm_2_1')(conv2)
  conv2 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_2_2')(norm2)
  norm2 = BatchNormalization(axis=4, name='norm_2_2')(conv2)
  pool2 = MaxPooling3D(pool_size=pool_size, name='pool_2')(norm2)

  conv3 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_3_1')(pool2)
  norm3 = BatchNormalization(axis=4, name='norm_3_1')(conv3)
  conv3 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_3_2')(norm3)
  norm3 = BatchNormalization(axis=4, name='norm_3_2')(conv3)
  pool3 = MaxPooling3D(pool_size=pool_size, name='pool_3')(norm3)

  conv4 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_4_1')(pool3)
  norm4 = BatchNormalization(axis=4, name='norm_4_1')(conv4)
  conv4 = Conv3D(512, conv_size, activation='relu', padding='same', name='conv_4_2')(norm4)
  norm4 = BatchNormalization(axis=4, name='norm_4_2')(conv4)
  pool4 = MaxPooling3D(pool_size=pool_size, name='pool_4')(norm4)

  conv5 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_5_1')(pool4)
  norm5 = BatchNormalization(axis=4, name='norm_5_1')(conv5)
  conv5 = Conv3D(512, conv_size, activation='relu', padding='same', name='conv_5_2')(norm5)
  norm5 = BatchNormalization(axis=4, name='norm_5_2')(conv5)

  up6 = UpSampling3D(size=pool_size, name='up_6')(norm5)
  up6 = concatenate([up6, norm4], axis=4, name='conc_6')
  drop6 = Dropout(rate=drop_out, name='drop_6')(up6)
  conv6 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_6_1')(drop6)
  conv6 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_6_2')(conv6)

  up7 = UpSampling3D(size=pool_size, name='up_7')(conv6)
  up7 = concatenate([up7, norm3], axis=4, name='conc_7')
  drop7 = Dropout(rate=drop_out, name='drop_7')(up7)
  conv7 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_7_1')(drop7)
  conv7 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_7_2')(conv7)

  up8 = UpSampling3D(size=pool_size, name='up_8')(conv7)
  up8 = concatenate([up8, norm2], axis=4, name='conc_8')
  drop8 = Dropout(rate=drop_out, name='drop_8')(up8)
  conv8 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_8_1')(drop8)
  conv8 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_8_2')(conv8)

  up9 = UpSampling3D(size=pool_size, name='up_9')(conv8)
  up9 = concatenate([up9, norm1], axis=4, name='conc_9')
  drop9 = Dropout(rate=drop_out, name='drop_9')(up9)
  conv9 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_9_1')(drop9)
  conv9 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_9_2')(conv9)

  conv10 = Conv3D(1, (1, 1, 1), name='conv_10')(conv9)
  act = Activation('sigmoid', name='act')(conv10)

  if mgpu == 1:
    print 'Compiling single GPU model'
    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss,
                  metrics=[dice_coef])
    return model
  elif mgpu > 1:
    print 'Compiling multi GPU model'
    model = Model(inputs=inputs, outputs=act)
    parallel_model = multi_gpu_model(model, gpus=mgpu)
    parallel_model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss,
                           metrics=[dice_coef])
    return parallel_model
  else:
    print 'ERROR Wrong number of GPUs defined'
    return
