"""
  ----------------------------------------
     HeartLoc - DeepCAC pipeline step1
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 2.7.17
  ----------------------------------------
  
"""

import os, multiprocessing, sys, glob
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import measurements


def run_core(patient):
  nrrdReader = sitk.ImageFileReader()
  nrrdWriter = sitk.ImageFileWriter()

  patientID = patient[0]
  print "Processing patient", patientID

  try:
    nrrdReader.SetFileName(patient[1])
    img_SITK_True_RAW = nrrdReader.Execute()
  except Exception as e:
    print 'Exception thrown when reading patient_NRRD_True_RAW_file:', patient[1], e
    return

  try:
    nrrdReader.SetFileName(patient[3])
    img_SITK_True_112 = nrrdReader.Execute()
  except Exception as e:
    print 'Exception thrown when reading patient_NRRD_True_112_file:', patient[3], e
    return

  try:
    patient_NPY_Pred_112 = np.load(patient[4], allow_pickle=True)
  except Exception as e:
    print 'Exception thrown when reading patient_NPY_Pred_112_file', patient[4], e
    return

  msk_NPY_Pred_112 = patient_NPY_Pred_112[3]

  # Remove all but the biggest segmented volume
  msk_NPY_Pred_112[msk_NPY_Pred_112 > 0.9] = 1
  msk_NPY_Pred_112[msk_NPY_Pred_112 < 1] = 0
  maxVol = 0
  maxLabel = 0
  labels, numLabels = measurements.label(msk_NPY_Pred_112)
  for label in range(1, numLabels + 1):
    vol = np.count_nonzero(labels == label)
    if vol > maxVol:
      maxVol = vol
      maxLabel = label
  msk_NPY_Pred_112 = np.zeros(msk_NPY_Pred_112.shape)
  msk_NPY_Pred_112[labels == maxLabel] = 1

  # Upsample and upsize mask and save as nrrd for further computation
  patient_SITK_Pred_112 = sitk.GetImageFromArray(msk_NPY_Pred_112)
  patient_SITK_Pred_112.CopyInformation(img_SITK_True_112)

  upSize = img_SITK_True_RAW.GetSize()
  upSpacing = img_SITK_True_RAW.GetSpacing()

  resFilter = sitk.ResampleImageFilter()
  msk_SITK_Pred_512 = resFilter.Execute(patient_SITK_Pred_112,
                                        upSize,
                                        sitk.Transform(),
                                        sitk.sitkNearestNeighbor,
                                        img_SITK_True_RAW.GetOrigin(),
                                        upSpacing,
                                        img_SITK_True_RAW.GetDirection(),
                                        0,
                                        img_SITK_True_RAW.GetPixelIDValue())

  msk_NPY_Pred_512 = sitk.GetArrayFromImage(msk_SITK_Pred_512)
  if np.sum(msk_NPY_Pred_512) == 0:
    print 'WARNING: Found emtpy mask for patient', patientID

  mask = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
          [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
          [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]

  # Remove all but the biggest segmented volume again
  msk_NPY_Pred_512[msk_NPY_Pred_512 > 0.9] = 1
  msk_NPY_Pred_512[msk_NPY_Pred_512 < 1] = 0
  maxVol = 0
  maxLabel = 0
  labels, numLabels = measurements.label(msk_NPY_Pred_512, structure=mask)
  if numLabels > 0:
    for objectNr in range(1, numLabels + 1):
      # vol = np.count_nonzero(labels == objectNr)
      vol = np.sum(msk_NPY_Pred_512[labels == objectNr])
      if vol > maxVol:
        maxVol = vol
        maxLabel = objectNr
    msk_NPY_Pred_512[labels == maxLabel] = 1

  msk_SITK_Pred_512 = sitk.GetImageFromArray(msk_NPY_Pred_512)
  msk_SITK_Pred_512.CopyInformation(img_SITK_True_RAW)

  nrrdWriter.SetFileName(patient[5])
  nrrdWriter.SetUseCompression(True)
  nrrdWriter.Execute(msk_SITK_Pred_512)


def upsample_results(curated_dir_path, resampled_dir_path, model_output_dir_path, model_output_nrrd_dir_path, num_cores):

  print "\nData upsampling:"

  THRESHOLD = 0.9
  pred_input = os.path.join(model_output_dir_path, 'npy')

  # read patient files
  patients = list()
  patient_NRRD_True_RAW_files = glob.glob(curated_dir_path + '/*_img.nrrd')
  for patient_NRRD_True_RAW_file in patient_NRRD_True_RAW_files:
    patientID = os.path.basename(patient_NRRD_True_RAW_file).replace('_img.nrrd', '')

    # Get all files
    msk_NRRD_True_RAW_file = patient_NRRD_True_RAW_file.replace('_img', '_msk')
    patient_NRRD_True_112_file = os.path.join(resampled_dir_path, patientID + '_img.nrrd')
    patient_NPY_Pred_112_file = os.path.join(pred_input, patientID + '_pred.npy')
    patient_SITK_Pred_512_file = os.path.join(model_output_nrrd_dir_path, patientID + '_pred.nrrd')

    if (not os.path.exists(patient_NRRD_True_RAW_file) or
        not os.path.exists(patient_NRRD_True_112_file) or
        not os.path.exists(patient_NPY_Pred_112_file)):
      print 'File doesnt exist', patientID
      print patient_NRRD_True_RAW_file
      print patient_NRRD_True_112_file
      print patient_NPY_Pred_112_file
      continue
    patients.append([patientID, patient_NRRD_True_RAW_file, msk_NRRD_True_RAW_file, patient_NRRD_True_112_file,
                     patient_NPY_Pred_112_file, patient_SITK_Pred_512_file])
  
  print 'Found', len(patients), 'patients under "%s"'%(curated_dir_path)

  # Process patients
  if num_cores == 1:
    for patient in patients:
      run_core(patient)
  elif num_cores > 1:
    pool = multiprocessing.Pool(processes=num_cores)
    pool.map(run_core, patients)
    pool.close()
    pool.join()
  else:
    print 'Wrong core number set in config file'
    sys.exit()
