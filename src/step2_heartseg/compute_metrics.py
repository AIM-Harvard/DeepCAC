"""
  ----------------------------------------
     HeartSeg - DeepCAC pipeline step2
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 2.7.17
  ----------------------------------------
  
"""

import os, sys, csv
from glob import glob
import numpy as np
import SimpleITK as sitk
from multiprocessing import Manager, Pool
from scipy.ndimage import measurements
from scipy.spatial import distance
from functools import partial


def diceCoef(y_true, y_pred, smooth=1.):
  y_true_f = y_true.flatten()
  y_pred_f = y_pred.flatten()
  intersection = np.sum(y_true_f * y_pred_f)
  dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
  return round(float(dice), 3)

## ----------------------------------------
## ----------------------------------------

def getCenter(cube):
  if np.sum(cube) > 0:
    onesCube = np.where(cube == 1)
    BB = [np.min(onesCube[0]), np.max(onesCube[0]),
                   np.min(onesCube[1]), np.max(onesCube[1]),
                   np.min(onesCube[2]), np.max(onesCube[2])]
    cen = [BB[0]+(BB[1]-BB[0])/2, BB[2]+(BB[3]-BB[2])/2, BB[4]+(BB[5]-BB[4])/2]
    return cen
  else:
    return [int(cube.shape[0]/2), int(cube.shape[1]/2), int(cube.shape[2]/2)]

## ----------------------------------------
## ----------------------------------------

def getCenterDiff(cube1, cube2):
  cen1 = getCenter(cube1)
  cen2 = getCenter(cube2)
  return round(distance.euclidean(cen1, cen2), 2)

## ----------------------------------------
## ----------------------------------------

def getCenterDiffMm(cube1, cube2, raw_spacing):
  cen1 = getCenter(cube1)
  cen2 = getCenter(cube2)
  cen1_mm = [cen1[0] * raw_spacing[2], cen1[1] * raw_spacing[1], cen1[2] * raw_spacing[0]]
  cen2_mm = [cen2[0] * raw_spacing[2], cen2[1] * raw_spacing[1], cen2[2] * raw_spacing[0]]
  return round(distance.euclidean(cen1_mm, cen2_mm), 0)

## ----------------------------------------
## ----------------------------------------

def getXYZdiff(cube1, cube2):
  cen1 = getCenter(cube1)
  cen2 = getCenter(cube2)
  return abs(cen1[2]-cen2[2]), abs(cen1[1]-cen2[1]), abs(cen1[0]-cen2[0])

## ----------------------------------------
## ----------------------------------------

def getXYZdiffMm(cube1, cube2, raw_spacing):
  cen1 = getCenter(cube1)
  cen2 = getCenter(cube2)
  return abs(cen1[2]-cen2[2])*raw_spacing[0], abs(cen1[1]-cen2[1])*raw_spacing[1], abs(cen1[0]-cen2[0])*raw_spacing[2]

## ----------------------------------------
## ----------------------------------------

def getComDiff(cube1, cube2):
  com1 = measurements.center_of_mass(cube1)
  com2 = measurements.center_of_mass(cube2)

  try:
    return round(distance.euclidean(com1, com2), 2)
  except Exception as e:
    return -1

## ----------------------------------------
## ----------------------------------------

def voxel2ccm(numVoxels, imgSpacing):
  return round(numVoxels * imgSpacing[0] * imgSpacing[1] * imgSpacing[2] / 1000, 2)

## ----------------------------------------
## ----------------------------------------

def runCore(raw_spacing, results_list, patientDict):
  nrrdReader = sitk.ImageFileReader()

  patientID = patientDict['PatientID']
  print 'Processing patient', patientID

  # LOAD DATA
  nrrdReader.SetFileName(patientDict['imgFile'])
  imgSitk = nrrdReader.Execute()
  imgCube = sitk.GetArrayFromImage(imgSitk)
  imgCube = np.asarray(imgCube, dtype=float)
  imgSpacing = imgSitk.GetSpacing()

  nrrdReader.SetFileName(patientDict['mskFile'])
  mskSitk = nrrdReader.Execute()
  mskCube = sitk.GetArrayFromImage(mskSitk)
  mskCube[mskCube == 2] = 0
  mskCube[mskCube > 0] = 1

  nrrdReader.SetFileName(patientDict['prdFile'])
  prdSitk = nrrdReader.Execute()
  prdCube = sitk.GetArrayFromImage(prdSitk)
  prdCube[prdCube>0.99] = 1
  prdCube[prdCube<1] = 0

  if np.sum(mskCube) == 0:
    print 'ERROR - Found empty manual mask for patient', patientID
    return

  if np.sum(prdCube) == 0:
    print 'ERROR - Found empty prd mask for patient', patientID
    return

  if not prdCube.shape == imgCube.shape:
    print 'Wrong size for patient', patientID, prdCube.shape
    return

  try:
    # 0) General data
    resultDict = {}
    resultDict['PatientID'] = patientID
    resultDict['MaxImgHu'] = round(np.max(imgCube), 2)

    # 1) Manual Mask
    # Heart volume
    resultDict['mskVox'] = np.sum(mskCube)
    resultDict['mskVol'] = voxel2ccm(resultDict['mskVox'], imgSpacing)

    # Heart mean, min, max density
    resultDict['mskMeanHu'] = round(float(np.sum(imgCube[np.where(mskCube == 1)])/np.sum(mskCube)), 2)
    resultDict['mskMinHu'] = round(np.min(imgCube[np.where(mskCube == 1)]), 2)
    resultDict['mskMaxHu'] = round(np.max(imgCube[np.where(mskCube == 1)]), 2)

    # 2) Predicted Mask
    # Heart volume
    resultDict['prdVox'] = np.sum(prdCube)
    resultDict['prdVol'] = voxel2ccm(resultDict['prdVox'], imgSpacing)

    # Heart mean, min, max density
    resultDict['prdMeanHu'] = round(float(np.sum(imgCube[np.where(prdCube == 1)])/np.sum(prdCube)), 2)
    resultDict['prdMinHu'] = round(np.min(imgCube[np.where(prdCube == 1)]), 2)
    resultDict['prdMaxHu'] = round(np.max(imgCube[np.where(prdCube == 1)]), 2)

    # 3) Comparison
    # Dice
    resultDict['diceHrt'] = diceCoef(mskCube, prdCube, 1.0)

    # Volume Diff
    resultDict['volHrtDiff'] = abs(round(resultDict['mskVol'] - resultDict['prdVol'], 2))

    # Geometric center difference
    resultDict['centerDiff'] = getCenterDiff(mskCube, prdCube)
    resultDict['centerDiff_X'], resultDict['centerDiff_Y'], resultDict['centerDiff_Z'] = \
      getXYZdiff(mskCube, prdCube)

    # Geometric center difference in mm
    resultDict['centerDiff_mm'] = getCenterDiffMm(mskCube, prdCube, raw_spacing)
    resultDict['centerDiff_X_mm'], resultDict['centerDiff_Y_mm'], resultDict['centerDiff_Z_mm'] = \
      getXYZdiffMm(mskCube, prdCube, raw_spacing)

    # Center of mass difference
    resultDict['comDiff'] = getComDiff(mskCube, prdCube)

    results_list.append(resultDict)
  except Exception as e:
    print 'EXCEPTION', patientID, e
    resultDict = {}
    resultDict['PatientID'] = patientID
    resultDict['MaxImgHu'] = 'NA'
    resultDict['mskVox'] = 'NA'
    resultDict['mskVol'] = 'NA'
    resultDict['mskMeanHu'] = 'NA'
    resultDict['mskMinHu'] = 'NA'
    resultDict['mskMaxHu'] = 'NA'
    resultDict['prdVox'] =  'NA'
    resultDict['prdVol'] =  'NA'
    resultDict['prdMeanHu'] =  'NA'
    resultDict['prdMinHu'] =  'NA'
    resultDict['prdMaxHu'] =  'NA'
    resultDict['diceHrt'] = '0'
    resultDict['volHrtDiff'] = 'NA'
    resultDict['centerDiff'] = 'NA'
    resultDict['centerDiff_X'] = 'NA'
    resultDict['centerDiff_Y'] = 'NA'
    resultDict['centerDiff_Z'] = 'NA'
    resultDict['comDiff'] = 'NA'
    resultDict['centerDiff_mm'] = 'NA'
    resultDict['centerDiff_X_mm'] = 'NA'
    resultDict['centerDiff_Y_mm'] = 'NA'
    resultDict['centerDiff_Z_mm'] = 'NA'
    results_list.append(resultDict)


def getFiles(patient_list, raw_dir, pred_dir, mask):
  prd_files = glob(pred_dir + '/*_pred.nrrd')

  for prd_file in prd_files:
    patient_dict = dict()
    patient_id = os.path.basename(prd_file).replace('_pred.nrrd', '')
    patient_dict['PatientID'] = patient_id

    imgFile = os.path.join(raw_dir, patient_id + '_img.nrrd')
    if not os.path.exists(imgFile):
      print 'ERROR: No image file found for patient', patient_id
      continue
    patient_dict['imgFile'] = imgFile

    if mask:
      msk_file = os.path.join(raw_dir, patient_id + '_msk.nrrd')
      if not os.path.exists(msk_file):
        print 'ERROR: No mask file found for patient', patient_id
        continue
      patient_dict['mskFile'] = msk_file

    patient_dict['prdFile'] = prd_file
    patient_list.append(patient_dict)

  print 'Found', len(patient_list), 'images under "%s"'%(pred_dir)
  return patient_list

## ----------------------------------------
## ----------------------------------------

def compute_metrics(cur_dir, pred_dir, output_dir, raw_spacing, num_cores, mask):

  print "\nComputing metrics:"
  print 'Loading patient data under "%s"'%(cur_dir)
  patient_list = []
  patient_list = getFiles(patient_list, cur_dir, pred_dir, mask)

  if num_cores == 1:
    results_list = []
    for patientDict in patient_list:
      runCore(raw_spacing, results_list, patientDict)
  elif num_cores > 1:
    with Manager() as manager:
      results_list = manager.list()
      pool = Pool(processes = num_cores)
      pool.map(partial(runCore, raw_spacing, results_list), patient_list)
      pool.close()
      pool.join()
      results_list = list(results_list)
  else:
    print 'Wrong core number set in config file'
    sys.exit()

  result_file = os.path.join(output_dir, 'results.csv')
  print 'Saving results to csv file under "%s"'%(result_file)

  with open(result_file, 'wb') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    title_row = ['PatientID', 'MaxImgHu', 'mskVox', 'mskVol', 'mskMeanHu', 'mskMinHu', 'mskMaxHu', 'prdVox', 'prdVol',
                 'prdMeanHu', 'prdMinHu', 'prdMaxHu', 'diceHrt', 'volHrtDiff', 'centerDiff', 'centerDiff_X',
                 'centerDiff_Y', 'centerDiff_Z', 'comDiff', 'centerDiff_mm', 'centerDiff_X_mm', 'centerDiff_Y_mm',
                 'centerDiff_Z_mm']
    filewriter.writerow(title_row)

    for resultDict in results_list:
      row = [resultDict['PatientID'],
             resultDict['MaxImgHu'],
             resultDict['mskVox'],
             resultDict['mskVol'],
             resultDict['mskMeanHu'],
             resultDict['mskMinHu'],
             resultDict['mskMaxHu'],
             resultDict['prdVox'],
             resultDict['prdVol'],
             resultDict['prdMeanHu'],
             resultDict['prdMinHu'],
             resultDict['prdMaxHu'],
             resultDict['diceHrt'],
             resultDict['volHrtDiff'],
             resultDict['centerDiff'],
             resultDict['centerDiff_X'],
             resultDict['centerDiff_Y'],
             resultDict['centerDiff_Z'],
             resultDict['comDiff'],
             resultDict['centerDiff_mm'],
             resultDict['centerDiff_X_mm'],
             resultDict['centerDiff_Y_mm'],
             resultDict['centerDiff_Z_mm']]
      filewriter.writerow(row)
