"""
  ----------------------------------------
     HeartSeg - DeepCAC pipeline step2
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 2.7.17
  ----------------------------------------
  
"""

import sys
import pickle
import os
import multiprocessing
from multiprocessing import  Manager
import numpy as np
import SimpleITK as sitk
from functools import partial


def expandSitk(imagesRawSitk, inter_size):
  oldSize = imagesRawSitk['rawImg'].GetSize()
  expSize = [oldSize[0], oldSize[1], inter_size[2]]

  sizeDif = [max(0, expSize[0] - oldSize[0]),
             max(0, expSize[1] - oldSize[1]),
             max(0, expSize[2] - oldSize[2])]

  newSizeDown = [0, 0, 0]
  newSizeUp = sizeDif

  padFilter = sitk.ConstantPadImageFilter()
  padFilter.SetPadUpperBound(newSizeUp)
  padFilter.SetPadLowerBound(newSizeDown)

  for key in imagesRawSitk:
    if 'Img' in key:
      air = -1024
    else:
      air = 0
    padFilter.SetConstant(air)
    imagesRawSitk[key] = padFilter.Execute(imagesRawSitk[key])

  return imagesRawSitk, imagesRawSitk['rawImg'].GetSize(), sizeDif, newSizeDown, newSizeUp

## ----------------------------------------
## ----------------------------------------

def cropSitk(patient_id, patient, imagesRawSitk, inter_size):
  oldSize = imagesRawSitk['rawImg'].GetSize()

  if 'upPredMskCen' in patient:
    mskCen = patient['upPredMskCen']
  elif 'upOrigMskCen' in patient:
    mskCen = patient['upOrigMskCen']
  else:
    print 'No mask center found for patient', patient_id
    return

  newSizeDown = [max(0, int(mskCen[2]-inter_size[0]/2)),
                 max(0, int(mskCen[1]-inter_size[1]/2)),
                 max(0, int(mskCen[0]-inter_size[2]/2))]
  newSizeDown = [min(newSizeDown[0], oldSize[0]-inter_size[0]),
                 min(newSizeDown[1], oldSize[1]-inter_size[1]),
                 min(newSizeDown[2], oldSize[2]-inter_size[2])]

  newSizeUp = [oldSize[0]-inter_size[0]-newSizeDown[0],
               oldSize[1]-inter_size[1]-newSizeDown[1],
               oldSize[2]-inter_size[2]-newSizeDown[2]]

  cropFilter = sitk.CropImageFilter()
  cropFilter.SetUpperBoundaryCropSize(newSizeUp)
  cropFilter.SetLowerBoundaryCropSize(newSizeDown)

  for key in imagesRawSitk:
    imagesRawSitk[key] = cropFilter.Execute(imagesRawSitk[key])

  newSize = imagesRawSitk['rawImg'].GetSize()
  sizeDif = [oldSize[0]-newSize[0], oldSize[1]-newSize[1], oldSize[2]-newSize[2]]

  return imagesRawSitk, newSize, sizeDif, newSizeDown, newSizeUp

## ----------------------------------------
## ----------------------------------------

def saveNrrd(patient_id, images_raw_sitk, output_dir, img_file):
  nrrd_writer = sitk.ImageFileWriter()
  nrrd_writer.SetFileName(img_file)
  nrrd_writer.SetUseCompression(True)
  nrrd_writer.Execute(images_raw_sitk['rawImg'])

  if 'rawMsk' in images_raw_sitk:
    msk_file = os.path.join(output_dir, patient_id + '_msk.nrrd')
    nrrd_writer.SetFileName(msk_file)
    nrrd_writer.SetUseCompression(True)
    nrrd_writer.Execute(images_raw_sitk['rawMsk'])

  if 'prdMsk' in images_raw_sitk:
    prd_file = os.path.join(output_dir, patient_id + '_prd.nrrd')
    nrrd_writer.SetFileName(prd_file)
    nrrd_writer.SetUseCompression(True)
    nrrd_writer.Execute(images_raw_sitk['prdMsk'])

## ----------------------------------------
## ----------------------------------------

def check_images(patient_id, imagesRawSitk, final_size, final_spacing):
  check = True

  for key in imagesRawSitk:
    if not tuple(np.round(final_size, 1)) == tuple(np.round(imagesRawSitk[key].GetSize(), 1)):
      check = False
      print 'Wrong size for patient', patient_id
      print tuple(np.round(final_size, 1)), tuple(np.round(imagesRawSitk[key].GetSize(), 1))

    if not tuple(np.round(final_spacing, 1)) == tuple(np.round(imagesRawSitk[key].GetSpacing(), 1)):
      check = False
      print 'Wrong spacing for patient', patient_id
      print tuple(np.round(final_spacing, 1)), tuple(np.round(imagesRawSitk[key].GetSpacing(), 1))

  return check

## ----------------------------------------
## ----------------------------------------

def getPatientFiles(patient, imagesRawSitk):
  nrrd_reader = sitk.ImageFileReader()
  # Read the raw image file
  nrrd_reader.SetFileName(patient['rawImg'])
  imagesRawSitk['rawImg'] = nrrd_reader.Execute()

  if 'rawMsk' in patient:
    # Read the raw mask file
    nrrd_reader.SetFileName(patient['rawMsk'])
    imagesRawSitk['rawMsk'] = nrrd_reader.Execute()

  if 'prdMsk' in patient:
    # Read the pred mask file
    nrrd_reader.SetFileName(patient['prdMsk'])
    imagesRawSitk['prdMsk'] = nrrd_reader.Execute()
  return imagesRawSitk

## ----------------------------------------
## ----------------------------------------

def downsampleSitk(imagesRawSitk, final_spacing, final_size):
  origSpacing = imagesRawSitk['rawImg'].GetSpacing()
  origSize = imagesRawSitk['rawImg'].GetSize()

  final_spacing[0] = origSize[0]*origSpacing[0]/final_size[0]
  final_spacing[1] = origSize[1]*origSpacing[1]/final_size[1]

  resFilter = sitk.ResampleImageFilter()

  for key in imagesRawSitk:
    if 'Img' in key:
      method = sitk.sitkLinear
    else:
      method = sitk.sitkNearestNeighbor
    imagesRawSitk[key] = resFilter.Execute(imagesRawSitk[key],
                                           final_size,
                                           sitk.Transform(),
                                           method,
                                           imagesRawSitk[key].GetOrigin(),
                                           final_spacing,
                                           imagesRawSitk[key].GetDirection(),
                                           0,
                                           imagesRawSitk[key].GetPixelIDValue())
  return imagesRawSitk

## ----------------------------------------
## ----------------------------------------

def runCore(patients, output_dir, diff_dict, final_size, final_spacing, inter_size, patient_id):
  patient = patients[patient_id]
  imgFile = os.path.join(output_dir, patient_id + '_img.nrrd')
  imagesRawSitk = {}

  imagesRawSitk = getPatientFiles(patient, imagesRawSitk)
  imagesRawSitk, expSize, sizeDifExpand, NSDexpt, NSUexp = expandSitk(imagesRawSitk, inter_size)
  imagesRawSitk, cropSize, sizeDifCrop, NSD_crop, NSUcrop = cropSitk(patient_id, patient, imagesRawSitk, inter_size)
  imagesRawSitk = downsampleSitk(imagesRawSitk, final_spacing, final_size)

  if not check_images(patient_id, imagesRawSitk, final_size, final_spacing):
    return

  saveNrrd(patient_id, imagesRawSitk, output_dir, imgFile)
  diff_dict[patient_id] = [sizeDifExpand, sizeDifCrop, NSDexpt, NSUexp, NSD_crop, NSUcrop]

  print 'Processing patient', patient_id

## ----------------------------------------
## ----------------------------------------

def crop_data(bb_calc_dir, output_dir, network_dir, inter_size, final_size, final_spacing, num_cores):
  print "\nData cropping:"
  
  pkl_input_file = os.path.join(bb_calc_dir, 'bbox.pkl')
  pkl_output_file = os.path.join(network_dir, 'diff_result.pkl')

  # Process the train data
  with open(pkl_input_file, "rb") as f:
    patients = pickle.load(f)
    print 'Found results for', len(patients), 'patients in "%s"'%(pkl_input_file)

    if num_cores == 1:
      diff_dict = {}
      for patient_key in patients.keys():
        runCore(patients, output_dir, diff_dict, final_size, final_spacing, inter_size, patient_key)
    elif num_cores > 1:
      with Manager() as manager:
        diff_dict = manager.dict()
        pool = multiprocessing.Pool(processes = num_cores)
        pool.map(partial(runCore, patients, output_dir, diff_dict, final_size, final_spacing, inter_size), patients)
        pool.close()
        pool.join()
        diff_dict = dict(diff_dict)
    else:
      print 'Wrong core number set in config file'
      sys.exit()

  print 'Saving', len(diff_dict), 'patients data to "%s"'%(pkl_output_file)
  with open(pkl_output_file, 'wb') as results_file:
    pickle.dump(diff_dict, results_file, pickle.HIGHEST_PROTOCOL)
