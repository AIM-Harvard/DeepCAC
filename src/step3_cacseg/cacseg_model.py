"""
  ----------------------------------------
     CAC segmentation - DL model arch.
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 2.7.17
  ----------------------------------------
  
"""

import os, math, pickle, sys

from functools import partial
from tensorflow.python.keras import utils
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import Input
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras.utils import *
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger, Callback, LearningRateScheduler


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

def getUnet3d(down_steps, input_shape, pool_size, conv_size, initial_learning_rate, mgpu,
              extended, drop_out, optimizer):

  if down_steps == 3:
    if extended:
      return getUnet3d_3_MGPU_extended(input_shape, pool_size=pool_size, conv_size=conv_size,
                                       initial_learning_rate=initial_learning_rate, mgpu=mgpu,
                                       drop_out=drop_out, optimizer=optimizer)
    else:
      return getUnet3d_3_MGPU(input_shape, pool_size=pool_size, conv_size=conv_size,
                              initial_learning_rate=initial_learning_rate, mgpu=mgpu)
  elif down_steps == 4:
    return getUnet3d_4_MGPU(input_shape, pool_size=pool_size, conv_size=conv_size,
                            initial_learning_rate=initial_learning_rate, mgpu=mgpu)
  else:
    print 'Wrong U-Net parameters specified ("down_steps")'

## ----------------------------------------
## ----------------------------------------

def getUnet3d_4_MGPU(input_shape, pool_size=(2, 2, 2), conv_size=(3, 3, 3),
                     initial_learning_rate=0.00001, mgpu=0):
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

  if mgpu > 1:
    model = Model(inputs=inputs, outputs=act)
    parallel_model = multi_gpu_model(model, gpus=mgpu)
    parallel_model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss,
                           metrics=[dice_coef])
    return parallel_model
  else:
    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss,
                  metrics=[dice_coef])
    return model

## ----------------------------------------
## ----------------------------------------

def getUnet3d_3_MGPU(input_shape, pool_size=(2, 2, 2), conv_size=(3, 3, 3),
                     initial_learning_rate=0.00001, mgpu=0):
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

  up5 = UpSampling3D(size=pool_size, name='up_5')(conv4)
  up5 = concatenate([up5, conv3], axis=4, name='conc_5')
  conv5 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_5_1')(up5)
  conv5 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_5_2')(conv5)

  up6 = UpSampling3D(size=pool_size, name='up_6')(conv5)
  up6 = concatenate([up6, conv2], axis=4, name='conc_6')
  conv6 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_6_1')(up6)
  conv6 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_6_2')(conv6)

  up7 = UpSampling3D(size=pool_size, name='up_7')(conv6)
  up7 = concatenate([up7, conv1], axis=4, name='conc_7')
  conv7 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_7_1')(up7)
  conv7 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_7_2')(conv7)

  conv8 = Conv3D(1, (1, 1, 1), name='conv_8')(conv7)
  act = Activation('sigmoid', name='act')(conv8)

  if mgpu > 1:
    model = Model(inputs=inputs, outputs=act)
    parallel_model = multi_gpu_model(model, gpus=mgpu)
    parallel_model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss,
                           metrics=[dice_coef])
    return parallel_model
  else:
    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss,
                  metrics=[dice_coef])
    return model

## ----------------------------------------
## ----------------------------------------

def getUnet3d_3_MGPU_extended(input_shape, pool_size=(2, 2, 2), conv_size=(3, 3, 3),
                              initial_learning_rate=0.00001, mgpu=0, drop_out=0.5,
                              optimizer='ADAM'):
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

  up5 = UpSampling3D(size=pool_size, name='up_5')(norm4)
  up5 = concatenate([up5, norm3], axis=4, name='conc_5')
  drop5 = Dropout(rate=drop_out, name='drop_5')(up5)
  conv5 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_5_1')(drop5)
  conv5 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_5_2')(conv5)

  up6 = UpSampling3D(size=pool_size, name='up_6')(conv5)
  up6 = concatenate([up6, norm2], axis=4, name='conc_6')
  drop6 = Dropout(rate=drop_out, name='drop_6')(up6)
  conv6 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_6_1')(drop6)
  conv6 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_6_2')(conv6)

  up7 = UpSampling3D(size=pool_size, name='up_7')(conv6)
  up7 = concatenate([up7, norm1], axis=4, name='conc_7')
  drop7 = Dropout(rate=drop_out, name='drop_7')(up7)
  conv7 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_7_1')(drop7)
  conv7 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_7_2')(conv7)

  conv8 = Conv3D(1, (1, 1, 1), name='conv_8')(conv7)
  act = Activation('sigmoid', name='act')(conv8)

  if mgpu > 1:
    model = Model(inputs=inputs, outputs=act)
    parallel_model = multi_gpu_model(model, gpus=mgpu)
    if optimizer == 'ADAM':
      parallel_model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss,
                             metrics=[dice_coef])
    elif optimizer == 'SGD':
      sgd = SGD(lr=initial_learning_rate, decay=0.01, momentum=0.5, nesterov=True)
      parallel_model.compile(optimizer=sgd, loss=dice_coef_loss,
                             metrics=[dice_coef])
    else:
      print('Wrong optimizer given')
      sys.exit()
    
    """
    print('---------------------------------------------------------------')
    print('Using model EXTENDED: DownSteps: 3 - optimizer:', optimizer, '- pool size:', pool_size,
          '- conv size:', conv_size, '- lr:', initial_learning_rate, '- mgpu:', mgpu,
          '- drop out:', drop_out)
    print('---------------------------------------------------------------')
    """
    return parallel_model
  else:
    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss,
                  metrics=[dice_coef])
    return model

## ----------------------------------------
## ----------------------------------------

# learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
  return initial_lrate * math.pow(drop, math.floor((1 + epoch) / float(epochs_drop)))

## ----------------------------------------
## ----------------------------------------

def pickle_dump(item, out_file):
  with open(out_file, "wb") as opened_file:
    pickle.dump(item, opened_file)

## ----------------------------------------
## ----------------------------------------

class SaveLossHistory(Callback):
  def on_train_begin(self, logs={}):
    self.losses = []

  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))
    pickle_dump(self.losses, "loss_history.pkl")

## ----------------------------------------
## ----------------------------------------

def get_callbacks(model_file, initial_learning_rate, learning_rate_drop, learning_rate_epochs,
                  logDir="."):
  model_checkpoint = ModelCheckpoint(model_file, save_best_only=True)
  logger = CSVLogger(os.path.join(logDir, "training.log"))
  history = SaveLossHistory()
  scheduler = LearningRateScheduler(partial(step_decay,
                                            initial_lrate=initial_learning_rate,
                                            drop=learning_rate_drop,
                                            epochs_drop=learning_rate_epochs))
  return [model_checkpoint, logger, history, scheduler]
