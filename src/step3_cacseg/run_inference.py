"""
  ----------------------------------------
     CAC segmentation - run DL inference
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 2.7.17
  ----------------------------------------
  
"""

import os, tables, socket, sys, pickle, math
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

from glob import glob
from skimage.io import imsave
from skimage.transform import resize
from scipy.ndimage import rotate, measurements

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose

import cacseg_model
from load_test_data import load_test_data


def getCubes(img, msk, cube_size):
  sizeX = img.shape[2]
  sizeY = img.shape[1]
  sizeZ = img.shape[0]

  cubeSizeX = cube_size[0]
  cubeSizeY = cube_size[1]
  cubeSizeZ = cube_size[2]

  n_z = int(math.ceil(float(sizeZ)/cubeSizeZ))
  n_y = int(math.ceil(float(sizeY)/cubeSizeY))
  n_x = int(math.ceil(float(sizeX)/cubeSizeX))

  sizeNew = [n_z*cubeSizeZ, n_y*cubeSizeY, n_x*cubeSizeX]

  imgNew = np.zeros(sizeNew, dtype=np.float16)
  imgNew[0:sizeZ, 0:sizeY, 0:sizeX] = img

  mskNew = np.zeros(sizeNew, dtype=np.int)
  mskNew[0:sizeZ, 0:sizeY, 0:sizeX] = msk

  n_ges = n_x * n_y * n_z
  n_4 = int(math.ceil(float(n_ges)/4.)*4)

  imgCubes = np.zeros((n_4, cubeSizeZ, cubeSizeY, cubeSizeX)) - 1  # -1 = air
  mskCubes = np.zeros((n_4, cubeSizeZ, cubeSizeY, cubeSizeX))

  count = 0
  for z in range(n_z):
    for y in range(n_y):
      for x in range(n_x):
        imgCubes[count] = imgNew[z*cubeSizeZ:(z+1)*cubeSizeZ,
                                 y*cubeSizeY:(y+1)*cubeSizeY,
                                 x*cubeSizeX:(x+1)*cubeSizeX]
        mskCubes[count] = mskNew[z*cubeSizeZ:(z+1)*cubeSizeZ,
                                 y*cubeSizeY:(y+1)*cubeSizeY,
                                 x*cubeSizeX:(x+1)*cubeSizeX]
        count += 1

  return imgCubes, mskCubes, n_x, n_y, n_z

## ----------------------------------------
## ----------------------------------------

def assemble(img, prd, n_x, n_y, n_z, cube_size):
  count = 0

  mskPredBig = np.zeros((n_z*img.shape[0], n_x*img.shape[1], n_x*img.shape[2]))
  mskPred = np.zeros(img.shape)

  # assemble cubes
  for z in range(n_z):
    for y in range(n_y):
      for x in range(n_x):
        mskPredBig[z*cube_size[2]:(z+1)*cube_size[2],
                   y*cube_size[1]:(y+1)*cube_size[1],
                   x*cube_size[0]:(x+1)*cube_size[0]] = prd[count,:,:,:,0]
        count += 1

  # crop cubes to original size
  mskPred = mskPredBig[0:mskPred.shape[0], 0:mskPred.shape[1], 0:mskPred.shape[2]]

  return mskPred

## ----------------------------------------
## ----------------------------------------

def export_png(patient_id, img, msk, prd, th, output_dir_png):

  prd[prd>th] = 1
  prd[prd<=th] = 0

  for z in range(len(img)):
    if np.sum(msk[z,:,:]) > 0 or np.sum(prd[z,:,:]) > 0:
      fig, ax = plt.subplots(1, 2, figsize=(32, 32))

      ax[0].imshow(img[z,:,:], cmap='gray')
      ax[0].imshow(msk[z,:,:], cmap='jet', alpha=0.4)

      ax[1].imshow(img[z,:,:], cmap='gray')
      ax[1].imshow(prd[z,:,:], cmap='jet', alpha=0.4)

      fileName = os.path.join(output_dir_png, patient_id + '_' + '_' + str(z) + '.png')
      plt.savefig(fileName)
      plt.close(fig)

## ----------------------------------------
## ----------------------------------------

def test(model, patient_data, cube_size, output_dir_npy, output_dir_png, th, export_cac_slices_png):
  patient_id = patient_data[0]
  print "Processing patient", patient_id
  img = patient_data[1]
  msk = patient_data[2]

  if img.shape[0] < cube_size[2] or img.shape[1] < cube_size[1] or img.shape[2] < cube_size[0]:
    print('Skipping patient', patient_id)
    return

  imgCubes, mskCubes, n_x, n_y, n_z = getCubes(img, msk, cube_size)

  prd_cubes = model.predict(imgCubes[:, :, :, :, np.newaxis])

  prd_msk = assemble(img, prd_cubes, n_x, n_y, n_z, cube_size)

  npy_file = os.path.join(output_dir_npy, patient_id + '_pred')
  np.save(npy_file, prd_msk)

  if export_cac_slices_png:
    export_png(patient_id = patient_id,
               img = img,
               msk = msk,
               prd = prd_msk,
               th = th,
               output_dir_png = output_dir_png)

## ----------------------------------------
## ----------------------------------------

def run_inference(data_dir, model_weights_dir_path, weights_file_name,
                  output_dir, export_cac_slices_png, has_manual_seg):
  
  print "\nDeep Learning model inference using 4xGPUs:" 
  
  # hard-coded model parameters
  th = 0.9
  smooth = 1.

  verbose = 0
  pool_size = (2, 2, 2)
  conv_size = (3, 3, 3)
  cube_size = [64, 64, 32]
  input_shape = (cube_size[2], cube_size[1], cube_size[0], 1)

  optimizer = 'ADAM'
  extended = True
  drop_out = 0.5
  mgpu = 4
  lr = 0.0001
  lr_drop = 0.7
  drop_epochs = 100
  batch_size = 20 * mgpu
  num_epochs = 600
  
  # number of the model downsampling steps 
  down_steps = 3  
  
  # set parameters for augmentation
  # training-time rotation
  rot_list = []
  max_queue_size = batch_size * mgpu
  workers = 24
  use_multiprocessing = True

  output_dir_npy = os.path.join(output_dir, 'npy')
  output_dir_png = os.path.join(output_dir, 'png')
  log_dir = os.path.join(output_dir, 'logs')
  
  if not os.path.exists(output_dir_npy): os.mkdir(output_dir_npy)
  if not os.path.exists(output_dir_png): os.mkdir(output_dir_png)
  if not os.path.exists(log_dir): os.mkdir(log_dir)

  weights_file = os.path.join(model_weights_dir_path, weights_file_name)

  test_data = load_test_data(data_dir, mask = has_manual_seg)
  print 'Found', len(test_data), 'patients under "%s"'%(data_dir)

  print 'Loading saved model from "%s"'%(weights_file)
  
  model = cacseg_model.getUnet3d(down_steps = down_steps, input_shape = input_shape, pool_size = pool_size,
                                 conv_size = conv_size, initial_learning_rate = lr, mgpu = mgpu,
                                 extended = extended, drop_out = drop_out, optimizer = optimizer)
  
  model.load_weights(weights_file)

  for patient_data in test_data:
    test(model = model,
         patient_data = patient_data,
         cube_size = cube_size,
         output_dir_npy = output_dir_npy,
         output_dir_png = output_dir_png,
         th = th,
         export_cac_slices_png = export_cac_slices_png)



