"""
  ----------------------------------------
     HeartSeg - DeepCAC pipeline step2
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 2.7.17
  ----------------------------------------
  
"""

import os
import sys
import pickle
from glob import glob
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import measurements
from functools import partial
from multiprocessing import Pool


def clean_sitk_mask(pred_sitk_512):
  pred_npy_512 = sitk.GetArrayFromImage(pred_sitk_512)
  mask = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
          [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
          [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]

  pred_npy_512[pred_npy_512 > 0.5] = 1
  pred_npy_512[pred_npy_512 < 1] = 0
  maxVol = 0
  maxLabel = 0
  labels, numLabels = measurements.label(pred_npy_512, structure=mask)
  if numLabels > 0:
    for objectNr in range(1, numLabels + 1):
      # vol = np.count_nonzero(labels == objectNr)
      vol = np.sum(pred_npy_512[labels == objectNr])

      if vol > maxVol:
        maxVol = vol
        maxLabel = objectNr

    pred_npy_512 = np.zeros(pred_npy_512.shape)
    pred_npy_512[labels == maxLabel] = 1
    pred_sitk_512_clean = sitk.GetImageFromArray(pred_npy_512)
    pred_sitk_512_clean.CopyInformation(pred_sitk_512)
    return pred_sitk_512_clean
  else:
    return pred_sitk_512

## ----------------------------------------
## ----------------------------------------

def expand_mask(patient_id, pkl_data, pred_sitk_288, imgSitk_512):
  
  # get size diff
  difData = pkl_data[patient_id]

  # expand x, y, z
  newSizeDown = difData[4]
  newSizeUp = difData[5]

  # correct for z dimension errors due to roundings
  # really not some code to be proud of - absolutely not
  transformedSizeX = pred_sitk_288.GetSize()[0] + newSizeDown[0] + newSizeUp[0] - int(difData[0][0])
  realSizeX = imgSitk_512.GetSize()[0]
  errorX = transformedSizeX - realSizeX

  transformedSizeY = pred_sitk_288.GetSize()[1] + newSizeDown[1] + newSizeUp[1] - int(difData[0][1])
  realSizeY = imgSitk_512.GetSize()[1]
  errorY = transformedSizeY - realSizeY

  transformedSizeZ = pred_sitk_288.GetSize()[2] + newSizeDown[2] + newSizeUp[2] - int(difData[0][2])
  realSizeZ = imgSitk_512.GetSize()[2]
  errorZ = transformedSizeZ - realSizeZ

  if newSizeDown[0] > newSizeUp[0]:
    newSizeDown[0] -= errorX
  else:
    newSizeUp[0] -= errorX

  if newSizeDown[1] > newSizeUp[1]:
    newSizeDown[1] -= errorY
  else:
    newSizeUp[1] -= errorY

  if newSizeDown[2] > newSizeUp[2]:
    newSizeDown[2] -= errorZ
  else:
    newSizeUp[2] -= errorZ

  padFilter = sitk.ConstantPadImageFilter()
  padFilter.SetPadUpperBound(newSizeUp)
  padFilter.SetPadLowerBound(newSizeDown)

  padFilter.SetConstant(-1)
  pred_sitk_512 = padFilter.Execute(pred_sitk_288)

  # Cut z
  zDif = int(difData[0][2])
  newSizeDown = [0, 0, 0]
  newSizeUp = [0, 0, zDif]

  cropFilter = sitk.CropImageFilter()
  cropFilter.SetUpperBoundaryCropSize(newSizeUp)
  cropFilter.SetLowerBoundaryCropSize(newSizeDown)
  pred_sitk_512 = cropFilter.Execute(pred_sitk_512)
  return pred_sitk_512

## ----------------------------------------
## ----------------------------------------

def run_core(cur_input, crop_input, output_dir, pkl_data, inter_size, patient):
  nrrd_reader = sitk.ImageFileReader()
  nrrd_writer = sitk.ImageFileWriter()

  patient_id = os.path.basename(patient).replace('_pred.npy', '')
  print 'Processing patient', patient_id

  # Read predicted mask
  pred_npy_128 = np.load(patient, allow_pickle=True)[3]
  pred_npy_128[pred_npy_128 > 0.5] = 1
  pred_npy_128[pred_npy_128 <= 0.5] = 0

  # Get original file - Needed for size correction
  img_true_file = os.path.join(cur_input, patient_id+'_img.nrrd')

  if not os.path.exists(img_true_file):
    print 'WARNING: Could not find image file for patient', patient_id
    return
  nrrd_reader.SetFileName(img_true_file)
  imgSitk_512 = nrrd_reader.Execute()

  # Convert to nrrd
  true_sitk_128_file = os.path.join(crop_input, patient_id+'_img.nrrd')
  nrrd_reader.SetFileName(true_sitk_128_file)
  img_sitk_128 = nrrd_reader.Execute()

  pred_sitk_128 = sitk.GetImageFromArray(pred_npy_128)
  pred_sitk_128.CopyInformation(img_sitk_128)

  interSpacing = imgSitk_512.GetSpacing()

  # upsample image
  resFilter = sitk.ResampleImageFilter()
  pred_sitk_288 = resFilter.Execute(pred_sitk_128, inter_size, sitk.Transform(), sitk.sitkLinear,
                                    pred_sitk_128.GetOrigin(), interSpacing, pred_sitk_128.GetDirection(),
                                    0, pred_sitk_128.GetPixelIDValue())

  # expand cropped mask to original size
  pred_sitk_512 = expand_mask(patient_id, pkl_data, pred_sitk_288, imgSitk_512)
  if not pred_sitk_512.GetSize()[2] == imgSitk_512.GetSize()[2]:
    print 'ERROR: Wrong upsampled size for patient'
    print patient_id, pred_sitk_512.GetSize()[2] - imgSitk_512.GetSize()[2]
    return

  # clean masks from small objects
  pred_sitk_512_clean = clean_sitk_mask(pred_sitk_512)

  output_file = os.path.join(output_dir, patient_id + '_pred.nrrd')
  nrrd_writer.SetFileName(output_file)
  nrrd_writer.SetUseCompression(True)
  nrrd_writer.Execute(pred_sitk_512_clean)

## ----------------------------------------
## ----------------------------------------

def upsample_results(cur_input, crop_input, network_dir, test_dir, output_dir, inter_size, num_cores):

  print "\nData upsampling:"  
  pkl_file = os.path.join(network_dir, 'diff_result.pkl')

  print 'Loading image data from "%s"'%(pkl_file)

  pkl_data = pickle.load(open(pkl_file, 'rb'))

  test_output = os.path.join(test_dir, 'npy')
  patients = glob(test_output + '/*')
  print 'Found', len(patients), 'patients under "%s"'%(test_output)

  if num_cores == 1:
    for patient in patients:
      run_core(cur_input, crop_input, output_dir, pkl_data, inter_size, patient)
  elif num_cores > 1:
    pool = Pool(processes = num_cores)
    pool.map(partial(run_core, cur_input, crop_input, output_dir, pkl_data, inter_size), patients)
    pool.close()
    pool.join()
  else:
    print 'Wrong core number set in config file'
    sys.exit()
