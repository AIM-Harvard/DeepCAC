"""
  ----------------------------------------
       CACSeg - DeepCAC pipeline step3
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 2.7.17
  ----------------------------------------
  
"""

import os
from glob import glob
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from multiprocessing import Pool
from functools import partial


def run_core(pred_dir, output_dir, dil_z_value, patient):
  nrrd_reader = sitk.ImageFileReader()
  nrrd_writer = sitk.ImageFileWriter()

  patient_id = (os.path.basename(patient)).split('_')[0]
  output_file = patient.replace(pred_dir, output_dir)

  print "Processing patient", patient_id

  # Read nrrd pred file and convert to npy
  nrrd_reader.SetFileName(patient)
  msk_pred_sitk = nrrd_reader.Execute()
  msk_pred_cube = sitk.GetArrayFromImage(msk_pred_sitk)

  if np.max(msk_pred_cube) == 0:
    print 'ERROR - Found emtpy mask for', patient_id
    return

  # Clean mask for dilation
  msk_pred_cube[msk_pred_cube >= 0.9] = 1
  msk_pred_cube[msk_pred_cube < 0.9] = 0

  # Dilate
  struct2 = np.zeros((3, 3, 3))
  struct2[0, 1, 1] = 1
  struct2[1, 1, 1] = 1
  struct2[2, 1, 1] = 1

  struct3 = np.zeros((3, 3, 3))
  struct3[0] = [[False, False, False],
                [False, True, False],
                [False, False, False]]
  struct3[1] = [[False, True, False],
                [True, True, True],
                [False, True, False]]
  struct3[2] = [[False, False, False],
                [False, True, False],
                [False, False, False]]

  msk_pred_cube_ext = ndimage.morphology.binary_dilation(msk_pred_cube,
                                                         structure = struct3,
                                                         iterations = dil_z_value
                                                        ).astype(msk_pred_cube.dtype)

  # Convert back to nrrd and save
  msk_pred_cube_ext_sitk = sitk.GetImageFromArray(msk_pred_cube_ext)
  msk_pred_cube_ext_sitk.CopyInformation(msk_pred_sitk)

  nrrd_writer.SetFileName(output_file)
  nrrd_writer.SetUseCompression(True)
  nrrd_writer.Execute(msk_pred_cube_ext_sitk)

## ----------------------------------------
## ----------------------------------------

def dilate_segmasks(pred_dir, output_dir, num_cores, dil_z_value = 11):
  
  print "Segmentation masks dilation:"

  patients = glob(pred_dir + '/*.nrrd')
  print 'Found', len(patients), 'patients under "%s"'%(pred_dir)

  pool = Pool(processes=num_cores)
  pool.map(partial(run_core, pred_dir, output_dir, dil_z_value), patients)
  pool.close()
  pool.join()






