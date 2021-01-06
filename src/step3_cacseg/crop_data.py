"""
  ----------------------------------------
       CACSeg - DeepCAC pipeline step3
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 2.7.17
  ----------------------------------------
  
"""

import matplotlib
import sys
import math
import os
import multiprocessing
from multiprocessing import Manager
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from glob import glob
from functools import partial


def save_png(patient_id, img_RAW_croped_cube, msk_RAW_croped_cube, png_output):
  size = img_RAW_croped_cube.shape

  fig, ax = plt.subplots(2, 3, figsize=(32, 16))

  ax[0, 0].imshow(img_RAW_croped_cube[int(size[0] / 2), :, :], cmap='gray')
  ax[0, 1].imshow(img_RAW_croped_cube[:, int(size[1] / 2), :], cmap='gray')
  ax[0, 2].imshow(img_RAW_croped_cube[:, :, int(size[2] / 2)], cmap='gray')

  ax[1, 0].imshow(msk_RAW_croped_cube[int(size[0] / 2), :, :], cmap='gray')
  ax[1, 1].imshow(msk_RAW_croped_cube[:, int(size[1] / 2), :], cmap='gray')
  ax[1, 2].imshow(msk_RAW_croped_cube[:, :, int(size[2] / 2)], cmap='gray')

  fileName = os.path.join(png_output, (str(patient_id) + ".png"))
  plt.savefig(fileName)
  plt.close(fig)

## ----------------------------------------
## ----------------------------------------

def run_core(raw_input, data_output, png_output, patch_size, has_manual_seg, export_png, prd_file):
  nrrdReader = sitk.ImageFileReader()

  patient_id = os.path.basename(prd_file).replace('_pred.nrrd', '')

  img3071file = os.path.join(data_output, patient_id + '_img_3071')
  imgFile = os.path.join(data_output, patient_id + '_img')
  mskFile = os.path.join(data_output, patient_id + '_msk')
  img_RAW_file = os.path.join(raw_input, patient_id + '_img.nrrd')

  # Read the raw image file
  if not os.path.exists(img_RAW_file):
    print 'ERROR - No image found for patient', patient_id
    return

  nrrdReader.SetFileName(img_RAW_file)
  img_RAW_sitk = nrrdReader.Execute()
  img_RAW_cube = sitk.GetArrayFromImage(img_RAW_sitk)

  # Read the predicted mask file
  nrrdReader.SetFileName(prd_file)
  msk_PRD_sitk = nrrdReader.Execute()
  msk_PRD_cube = sitk.GetArrayFromImage(msk_PRD_sitk)

  if not img_RAW_cube.shape == msk_PRD_cube.shape:
    print 'ERROR: Wrong dimesnion found for patient', patient_id
    print img_RAW_cube.shape, msk_PRD_cube.shape
    return

  if np.sum(msk_PRD_cube) == 0:
    print 'Found empty predicted mask for patient', patient_id
    return

  if has_manual_seg:
    msk_RAW_file = os.path.join(raw_input, patient_id + '_msk.nrrd')

    if not os.path.exists(msk_RAW_file):
      print 'ERROR - No mask file found for patient', patient_id
      return

    nrrdReader.SetFileName(msk_RAW_file)
    msk_RAW_sitk = nrrdReader.Execute()
    msk_RAW_cube = sitk.GetArrayFromImage(msk_RAW_sitk)

    if not img_RAW_cube.shape == msk_RAW_cube.shape:
      print 'ERROR: Wrong dimension found for patient', patient_id
      print img_RAW_cube.shape, msk_RAW_cube.shape, msk_PRD_cube.shape
      return

    if np.sum(msk_RAW_cube) > 0:
      msk_RAW_cube[msk_RAW_cube <= 2] = 0
      msk_RAW_cube[msk_RAW_cube > 0] = 1
    else:
      print 'Warning - Found empty manual mask for patient', patient_id

  msk_PRD_cube[msk_PRD_cube >= 0.5] = 1
  msk_PRD_cube[msk_PRD_cube < 0.5] = 0

  # Msk predicted heart in image
  img_HRT_cube = np.copy(img_RAW_cube)
  img_HRT_cube[msk_PRD_cube == 0] = -1024

  # Get BB
  msk_PRD_ones = np.where(msk_PRD_cube > 0)

  if np.sum(msk_PRD_ones) == 0:
    print 'ERROR: Found empty mask for patient', patient_id
    return

  hrt_BB = [np.min(msk_PRD_ones[0]), np.max(msk_PRD_ones[0]),
            np.min(msk_PRD_ones[1]), np.max(msk_PRD_ones[1]),
            np.min(msk_PRD_ones[2]), np.max(msk_PRD_ones[2])]

  # Make sure to crop to minimum size. This is to prevent the next steps to work correctly
  # although it indicates a wrong heart segmentation
  if (hrt_BB[1] - hrt_BB[0]) < patch_size[0]:
    offset = int(math.ceil((patch_size[0] - (hrt_BB[1] - hrt_BB[0])) / 2))
    hrt_BB[0] = max(0, hrt_BB[0] - offset)
    hrt_BB[1] = hrt_BB[0] + patch_size[0]

  if (hrt_BB[3] - hrt_BB[2]) < patch_size[1]:
    offset = int(math.ceil((patch_size[1] - (hrt_BB[3] - hrt_BB[2])) / 2))
    hrt_BB[2] = max(0, hrt_BB[2] - offset)
    hrt_BB[3] = hrt_BB[2] + patch_size[1]

  if (hrt_BB[5] - hrt_BB[4]) < patch_size[2]:
    offset = int(math.ceil((patch_size[2] - (hrt_BB[5] - hrt_BB[4])) / 2))
    hrt_BB[4] = max(0, hrt_BB[4] - offset)
    hrt_BB[5] = hrt_BB[4] + patch_size[2]

  print "Processing patient", patient_id

  # Crop files to heart
  img_RAW_croped_cube = img_HRT_cube[hrt_BB[0]:hrt_BB[1], hrt_BB[2]:hrt_BB[3], hrt_BB[4]:hrt_BB[5]]
  if has_manual_seg:
    msk_RAW_croped_cube = msk_RAW_cube[hrt_BB[0]:hrt_BB[1], hrt_BB[2]:hrt_BB[3], hrt_BB[4]:hrt_BB[5]]
  msk_PRD_croped_cube = msk_PRD_cube[hrt_BB[0]:hrt_BB[1], hrt_BB[2]:hrt_BB[3], hrt_BB[4]:hrt_BB[5]]

  np.save(img3071file, img_RAW_croped_cube)

  img_RAW_croped_cube = ((np.clip(img_RAW_croped_cube, -1024.0, 3071.0)) - 1023.5) / 2047.5
  if has_manual_seg:
    msk_RAW_croped_cube[msk_RAW_croped_cube != 1] = 0
    msk_RAW_croped_cube[msk_RAW_croped_cube != 1] = 0

  np.save(imgFile, img_RAW_croped_cube)
  if has_manual_seg: np.save(mskFile, msk_RAW_croped_cube)
  np.save(os.path.join(data_output, patient_id + '_prd'), msk_PRD_croped_cube)

  if export_png:
    if has_manual_seg:
      save_png(patient_id, img_RAW_croped_cube, msk_RAW_croped_cube, png_output)
    else:
      save_png(patient_id, img_RAW_croped_cube, np.zeros(img_RAW_croped_cube.shape), png_output)

## ----------------------------------------
## ----------------------------------------

def crop_data(raw_input, prd_input, data_output, png_output, patch_size, num_cores, has_manual_seg, export_png):

  print "\nHeart cropping:"

  prd_files = glob(prd_input + '/*.nrrd')
  print 'Found', len(prd_files), 'files under "%s"'%(prd_input)

  if num_cores == 1:
    for prd_file in prd_files:
      run_core(raw_input = raw_input,
               data_output = data_output,
               png_output = png_output,
               patch_size = patch_size,
               has_manual_seg = has_manual_seg,
               export_png = export_png,
               prd_file = prd_file)
      
  with Manager() as manager:
    pool = multiprocessing.Pool(processes=num_cores)
    pool.map(partial(run_core, raw_input,
                     data_output, png_output, patch_size, has_manual_seg, export_png), prd_files)
    pool.close()
    pool.join()
  # """

