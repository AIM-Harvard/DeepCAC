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
import multiprocessing
import numpy as np
import SimpleITK as sitk
from glob import glob
from scipy.spatial import distance
from multiprocessing import Manager
from functools import partial


def get_files_for_patient(patients_dict, pred_dir, img_file, has_manual_seg, run):
  
  """
  Read patient files information and store/return them in a dictionary.

  @params:
    patients_dict  - required : dictionary storing patients' information to be returned
    pred_dir       - required : directory storing inferred NRRD file(s) (CT and segmask) from step1
    img_file       - required : NRRD file storing the CT for a given patient
    has_manual_seg - required : whether a manual segmentation for the volume is available or not
    run            - required : whether the code is being run for training or testing purposes
    
  """
  
  patient = dict()
  patient_id = os.path.basename(img_file).replace('_img.nrrd', '')
  patient['rawImg'] = img_file

  if has_manual_seg:
    raw_msk_file = img_file.replace('img', 'msk')
    if os.path.exists(raw_msk_file):
      patient['rawMsk'] = raw_msk_file
    else:
      print 'No mask file found for patient', patient_id
      return

  if run == 'Test':
    prd_msk_file = os.path.join(pred_dir, patient_id + '_pred.nrrd')
    if os.path.exists(prd_msk_file):
      patient['prdMsk'] = prd_msk_file
    else:
      print 'No pred file found for patient', patient_id
      return

  patients_dict[patient_id] = patient

## ----------------------------------------
## ----------------------------------------

def get_msk_cube(msk_sitk_file):
  
  """
  Get numpy arrays of the segmentation masks (loaded as SimpleITK objects).

  @params:
    msk_sitk_file - required : path to the segmask to be loaded using SimpleITK
    
  """
  
  nrrd_reader = sitk.ImageFileReader()
  nrrd_reader.SetFileName(msk_sitk_file)
  msk_sitk = nrrd_reader.Execute()
  msk_cube = sitk.GetArrayFromImage(msk_sitk)
  msk_cube[msk_cube == 2] = 0
  msk_cube[msk_cube > 0] = 1
  return msk_cube

## ----------------------------------------
## ----------------------------------------

def get_bb_center(patient_id, cube):
  
  """
  Compute the bounding box center, handling some exceptions for wrong segmentation masks.

  @params:
    patient_id - required : ID of the patient
    cube       - required : mask to predict the bounding box center of
    
  """
  
  # check for obviously wrong masks (too small or too big)
  cube_sum = np.sum(cube)
  if cube_sum < 100000 or cube_sum > 999999:
    # set bbox to whole image - bb center equals the image center
    msk_bb = [0, len(cube), 0, len(cube[0]), 0, len(cube[0][0])]
  else:
    msk_ones = np.where(cube == 1)
    
    if len(msk_ones[0]) == 0:
      print 'ERROR orig:', patient_id
      return
      
    msk_bb = [np.min(msk_ones[0]), np.max(msk_ones[0]), np.min(msk_ones[1]), np.max(msk_ones[1]),
              np.min(msk_ones[2]), np.max(msk_ones[2])]

  msk_cen = [msk_bb[0] + (msk_bb[1] - msk_bb[0]) / 2, msk_bb[2] + (msk_bb[3] - msk_bb[2]) / 2,
             msk_bb[4] + (msk_bb[5] - msk_bb[4]) / 2]

  return msk_bb, msk_cen

## ----------------------------------------
## ----------------------------------------

def run_core(patients, result_dict, patient_id):
  
  """
  Core function for the bbox computation, to be run (potentially) with multiprocessing.

  @params:
    patients    - required : dictionary storing basic patients (files) information
    result_dict - required : dictionary to be returned upon completion of the core function
    patient_id  - required : ID of the patient
    
  """
  
  patient = patients[patient_id]

  if 'rawMsk' in patient.keys():
    # get true mask, bbox and center of bbox
    up_orig_msk_cube = get_msk_cube(patient['rawMsk'])    
    if up_orig_msk_cube.size == 0:
      return
    patient['upOrigMskBB'], patient['upOrigMskCen'] = get_bb_center(patient_id = patient_id,
                                                                    cube = up_orig_msk_cube)
    patient['size'] = up_orig_msk_cube.shape

  if 'prdMsk' in patient.keys():
    # get pred mask, bbox and center of bbox
    up_pred_msk_cube = get_msk_cube(patient['prdMsk'])
    patient['upPredMskBB'], patient['upPredMskCen'] = get_bb_center(patient_id = patient_id,
                                                                    cube = up_pred_msk_cube)
    if 'size' not in patient.keys():
      patient['size'] = up_pred_msk_cube.shape

  if 'upOrigMskCen' in patient.keys() and 'upPredMskCen' in patient.keys():
    # compute the distance of bbox centers
    patient['distCEN'] = distance.euclidean(patient['upOrigMskCen'], patient['upPredMskCen'])

  result_dict[patient_id] = patient

## ----------------------------------------
## ----------------------------------------

def compute_bbox(cur_dir, pred_dir, output_dir, num_cores, has_manual_seg, run):
  
  """
  Compute bounding box for the segmentation masks.

  @params:
    cur_dir        - required : directory storing curated NRRD file(s) (CT and segmask) from step1
    pred_dir       - required : directory storing inferred NRRD file(s) (CT and segmask) from step1
    output_dir     - required : output directory for the pkl file storing bbox information
    num_cores      - required : number of cores to use for the multiprocessing 
    has_manual_seg - required : whether a manual segmentation for the volume is available or not
    run            - required : whether the code is being run for training or testing purposes
    
  """
  
  print "Bounding box calculation:"
  
  img_files = glob(cur_dir + '/*_img.nrrd')
  
  print 'Found', len(img_files), 'images under "%s"'%(cur_dir)

  patients_dict = dict()
  for img_file in img_files:
    get_files_for_patient(patients_dict = patients_dict,
                          pred_dir = pred_dir,
                          img_file = img_file,
                          has_manual_seg = has_manual_seg,
                          run = run)
  
  print 'Loading the masks inferred on step1 for', len(patients_dict), 'patients under "%s"'%(pred_dir)

  if num_cores == 1:
    result_dict = dict()
    for patient_id in patients_dict:
      run_core(patients = patients_dict,
               result_dict = result_dict,
               patient_id = patient_id)
      
  elif num_cores > 1:
    # Process patients on all cpu cores
    with Manager() as manager:
      result_dict = manager.dict()
      pool = multiprocessing.Pool(processes = num_cores)
      pool.map(partial(run_core, patients_dict, result_dict), patients_dict.keys())
      pool.close()
      pool.join()
      result_dict = dict(result_dict)
  else:
    print 'Wrong core number set in config file'
    sys.exit()

  results_file_name = os.path.join(output_dir, 'bbox.pkl')
  with open(results_file_name, 'wb') as results_file:
    pickle.dump(result_dict, results_file, pickle.HIGHEST_PROTOCOL)
