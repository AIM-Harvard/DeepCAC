
"""
  ----------------------------------------
     HeartLoc - DeepCAC pipeline step1
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 2.7.17
  ----------------------------------------
  

  Template script for preparing the input data to the deep learning pipeline
  
  IMPORTANT: This script needs to be adapted for different data sets. 
  The outputs are two *.nrrd files: the preprocessed CT volume and, if available,
  the manual segmentation mask preprocessed using the same parameters as the former.
  
  N.B. either all the patients have an associated (manual) segmentation mask, or none should.
  If the dataset in question is half and half, consider dividing it in two different sub-cohorts
  (and then run the pipeline on both, changing just "has_manual_seg" in the config file).
  
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from glob import glob
from functools import partial
from multiprocessing import Pool


def resample_sitk(img_sitk, method, curated_spacing, curated_size = None):

  """
  Resample volumes exploiting Simple ITK. 

  @params:
    img_sitk        - required : SimpleITK image, resulting from sitk.ImageFileReader().Execute()
    method          - required : SimpleITK interpolation method (e.g., sitk.sitkLinear)
    curated_spacing - required : list containing the desired spacing, in mm, of the output data
    curated_size    - optional : size (in voxels) of the output volume (defaults to None,
                                 computed from img_sitk and the desired spacing curated_spacing)
  """

  orig_spacing = tuple(np.round(img_sitk.GetSpacing(), 2))
  orig_size = img_sitk.GetSize()

  # if curated_size is not specified, compute it starting from the spacing and the size of the 
  # volume to resample, and the desired "curated_spacing"
  if curated_size is None:
    curated_size = [int(orig_size[0] * orig_spacing[0] / curated_spacing[0]),
                    int(orig_size[1] * orig_spacing[1] / curated_spacing[1]),
                    int(orig_size[2] * orig_spacing[2] / curated_spacing[2])]

  res_filter = sitk.ResampleImageFilter()

  """
  sitk.ResampleImageFilter() arguments:
    - SimpleITK image
    - Output image size
    - Transform, interface to ITK transform objects to be used wiht ResampleImageFilter
    - Interpolation method
    - Output image origin
    - Output image spacing
    - Output image direction
    - Default pixel value
    - Output pixel type
  """

  img_sitk = res_filter.Execute(img_sitk, curated_size, sitk.Transform(), method, img_sitk.GetOrigin(),
                                curated_spacing, img_sitk.GetDirection(), 0, img_sitk.GetPixelIDValue())

  return img_sitk, curated_size


## ----------------------------------------
## ----------------------------------------

def expand_sitk(img_sitk, curated_size, pad_value, new_size_up = None, new_size_down = None):
  
  """
  In-plane volumes padding exploiting Simple ITK. Does not pad in the last dimension (along axial).
  Furthermore, this will maintain correct physical location of the image.

  @params:
    img_sitk      - required : SimpleITK image, resulting from sitk.ImageFileReader().Execute()
    curated_size  - required : list containing the desired size, in voxels, of the output data
    pad_value     - required : constant value used to pad the image
    new_size_up   - optional : voxels/pixels to add to the image upper sides
                               (defaults to None, computed from img_sitk and curated_size)
    new_size_down - optional : voxels/pixels to add to the image lower sides
                               (defaults to None, computed from img_sitk and curated_size)
  """
  
  if new_size_up is None:
    
    # get the size of the image to be processed
    new_size = img_sitk.GetSize()
    
    # define how many voxels/pixels will be added to the image lower sides ("before" the data).
    new_size_down = [max(0, int((curated_size[0] - new_size[0]) / 2)),
                     max(0, int((curated_size[1] - new_size[1]) / 2)),
                     0]
    
    # define how many voxels/pixels will be added to the image upper sides ("after" the data).
    new_size_up = [max(0, curated_size[0] - new_size[0] - new_size_down[0]),
                   max(0, curated_size[1] - new_size[1] - new_size_down[1]),
                   0]

  pad_filter = sitk.ConstantPadImageFilter()
  
  pad_filter.SetConstant(pad_value)
  pad_filter.SetPadUpperBound(new_size_up)
  pad_filter.SetPadLowerBound(new_size_down)
  
  img_sitk = pad_filter.Execute(img_sitk)

  return img_sitk, new_size_up, new_size_down


## ----------------------------------------
## ----------------------------------------

def crop_sitk(img_sitk, curated_size, new_size_up = None, new_size_down = None):
  
    
  """
  In-plane volumes cropping exploiting Simple ITK. Does not crop in the last dimension (along axial).
  Furthermore, this will maintain correct physical location of the image.

  @params:
    img_sitk      - required : SimpleITK image, resulting from sitk.ImageFileReader().Execute()
    curated_size  - required : list containing the desired size, in voxels, of the output data
    new_size_up   - optional : voxels/pixels to add to the image upper sides
                               (defaults to None, computed from img_sitk and curated_size)
    new_size_down - optional : voxels/pixels to add to the image lower sides
                               (defaults to None, computed from img_sitk and curated_size)
  """
  
  if new_size_up is None:
    
    # get the size of the image to be processed
    old_size = img_sitk.GetSize()
    
    # define how many voxels/pixels will be cropped from the image lower sides ("before" the data).
    new_size_down = [max(0, int((old_size[0] - curated_size[0]) / 2)),
                     max(0, int((old_size[1] - curated_size[1]) / 2)),
                     0]
                     
    # define how many voxels/pixels will be cropped from the image upper sides ("after" the data).
    new_size_up = [max(0, old_size[0] - curated_size[0] - new_size_down[0]),
                   max(0, old_size[1] - curated_size[1] - new_size_down[1]),
                   0]

  crop_filter = sitk.CropImageFilter()
  
  crop_filter.SetUpperBoundaryCropSize(new_size_up)
  crop_filter.SetLowerBoundaryCropSize(new_size_down)
  
  img_sitk = crop_filter.Execute(img_sitk)

  return img_sitk, new_size_up, new_size_down


## ----------------------------------------
## ----------------------------------------

def check_img(patient_id, img_sitk, curated_size, curated_spacing):
  
  """
  Check if the given Simple ITK image matches a given size and spacing.

  @params:
    patient_id      - required : patient ID
    img_sitk        - required : SimpleITK image, resulting from sitk.ImageFileReader().Execute()
    curated_size    - required : size, in voxels, that img_sitk should match
    curated_spacing - required : spacing, in mm, that img_sitk should match
  
  """
  
  # check size
  if not img_sitk.GetSize()[0] == curated_size[0] or not img_sitk.GetSize()[1] == curated_size[1]:
    print 'Error, wrong image size', patient_id, img_sitk.GetSize(), img_sitk.GetSize()
    return False

  # check spacing
  if(not round(img_sitk.GetSpacing()[0], 2) == curated_spacing[0] or
     not round(img_sitk.GetSpacing()[1], 2) == curated_spacing[1] or
     not round(img_sitk.GetSpacing()[2], 2) == curated_spacing[2]):
    print 'Error, wrong image spacing', patient_id, np.round(img_sitk.GetSpacing(), 2)
    return False

  return True


## ----------------------------------------
## ----------------------------------------

def check_mask(patient_id, img_sitk, msk_sitk):
  
  """
  Check if the given CT volume matches in size and spacing the corresponding segmentation mask.

  @params:
    patient_id  - required : patient ID
    img_sitk    - required : SimpleITK image (CT), resulting from sitk.ImageFileReader().Execute()
    msk_sitk    - required : SimpleITK image (segmask), resulting from sitk.ImageFileReader().Execute()
  
  """
  
  # check size
  if(not img_sitk.GetSize()[0] == msk_sitk.GetSize()[0] or
     not img_sitk.GetSize()[1] == msk_sitk.GetSize()[1] or
     not img_sitk.GetSize()[2] == msk_sitk.GetSize()[2]):
    print 'Error, wrong mask size', patient_id, img_sitk.GetSize(), msk_sitk.GetSize()
    return False

  # check spacing
  if(not round(img_sitk.GetSpacing()[0], 2) == round(msk_sitk.GetSpacing()[0], 2) or
     not round(img_sitk.GetSpacing()[1], 2) == round(msk_sitk.GetSpacing()[1], 2) or
     not round(img_sitk.GetSpacing()[2], 2) == round(msk_sitk.GetSpacing()[2], 2)):
    print 'Error, wrong mask spacing', patient_id, np.round(img_sitk.GetSpacing(), 2), np.round(msk_sitk.GetSpacing(), 2)
    return False
  return True


## ----------------------------------------
## ----------------------------------------

def plot_sitk(patient_id, img_sitk, qc_curated_dir_path):
  
  """
  Quality control - for the given volume, outputs a figure containing the three center slices
  (for the CT volume, one for each of the main view).

  @params:
    patient_id          - required : patient ID
    img_sitk            - required : SimpleITK image (CT), resulting from sitk.ImageFileReader().Execute()
    qc_curated_dir_path - required : output directory for the png file
  
  """
  
  png_file = os.path.join(qc_curated_dir_path, patient_id + 'img.png')
  img_cube = sitk.GetArrayFromImage(img_sitk)

  fig, ax = plt.subplots(1, 3, figsize = (32, 8))
  ax[0].imshow(img_cube[int(img_cube.shape[0]/2), :, :], cmap = 'gray')
  ax[1].imshow(img_cube[:, int(img_cube.shape[1]/2), :], cmap = 'gray')
  ax[2].imshow(img_cube[:, :, int(img_cube.shape[2]/2)], cmap = 'gray')

  plt.savefig(png_file, bbox_inches = 'tight')
  plt.close(fig)


## ----------------------------------------
## ----------------------------------------

def plot_sitk_msk(patient_id, img_sitk, msk_sitk, qc_curated_dir_path):
  
  """
  Quality control - for the given volume, outputs a figure containing the three center slices
  (for the CT volume, one for each of the main view) with the segmentation mask overlapped.

  @params:
    patient_id          - required : patient ID
    img_sitk            - required : SimpleITK image (CT), resulting from sitk.ImageFileReader().Execute()
    msk_sitk            - required : SimpleITK image (segmask), resulting from sitk.ImageFileReader().Execute()
    qc_curated_dir_path - required : output directory for the png file
  
  """
  
  png_file = os.path.join(qc_curated_dir_path, patient_id + 'img.png')
  img_cube = sitk.GetArrayFromImage(img_sitk)
  msk_cube = sitk.GetArrayFromImage(msk_sitk)

  fig, ax = plt.subplots(1, 3, figsize = (32, 8))
  ax[0].imshow(img_cube[int(img_cube.shape[0]/2), :, :], cmap = 'gray')
  ax[1].imshow(img_cube[:, int(img_cube.shape[1]/2), :], cmap = 'gray')
  ax[2].imshow(img_cube[:, :, int(img_cube.shape[2]/2)], cmap = 'gray')

  ax[0].imshow(msk_cube[int(img_cube.shape[0]/2), :, :], cmap = 'jet', alpha = 0.4)
  ax[1].imshow(msk_cube[:, int(img_cube.shape[1]/2), :], cmap = 'jet', alpha = 0.4)
  ax[2].imshow(msk_cube[:, :, int(img_cube.shape[2]/2)], cmap = 'jet', alpha = 0.4)

  plt.savefig(png_file, bbox_inches = 'tight')
  plt.close(fig)


## ----------------------------------------
## ----------------------------------------

# Process a single scan on one CPU core
def run_core(curated_dir_path, qc_curated_dir_path, export_png,
             has_manual_seg, curated_size, curated_spacing, patients_data, patient_id):

    
  """
  Pre-processing first step core function to be run (potentially) with multiprocessing.

  @params:

    curated_dir_path    - required : output directory for the file(s) (CT and segmask)
    qc_curated_dir_path - required : output directory for the png file
    export_png          - required : whether to export the quality control png or not
    has_manual_seg      - required : whether a manual segmentation for the volume is available or not
    curated_size            - required : list containing the desired size, in voxels, of the output data
    curated_spacing         - required : list containing the desired spacing, in mm, of the output data
    patients_data       - required : (pointer to) dictionary storing, for each patient, a list
                                     containing the path to the CT and segmask NRRD files
    patient_id          - required : patient ID, used to index "patients_data"

  """

  print 'Processing patient', patient_id
  
  # init SITK reader and writer, load the CT volume in a SITK object
  nrrd_reader = sitk.ImageFileReader()
  nrrd_writer = sitk.ImageFileWriter()
  nrrd_reader.SetFileName(patients_data[patient_id][0])
  img_sitk = nrrd_reader.Execute()
  
  
  # take care of the size/spacing difference - resample SITK image, expand/crop
  img_sitk, curated_size = resample_sitk(img_sitk = img_sitk, 
                                         method = sitk.sitkLinear,
                                         curated_spacing = curated_spacing)
  
  img_sitk, img_exp_up, img_exp_dn = expand_sitk(img_sitk = img_sitk, 
                                                 curated_size = curated_size,
                                                 pad_value = -1024)
  
  img_sitk, img_crp_up, img_crp_dn = crop_sitk(img_sitk = img_sitk, curated_size = curated_size)
  
  # if the check on the CT fails, return
  if not check_img(patient_id, img_sitk, curated_size, curated_spacing):
    return

  # save preprocessed CT volume  
  nrrd_writer.SetFileName(os.path.join(curated_dir_path, patient_id + '_img.nrrd'))
  nrrd_writer.SetUseCompression(True)
  nrrd_writer.Execute(img_sitk)

  # if the segmask is not available but export_png is set to True, export the CT quality control png
  if export_png and not has_manual_seg:
    plot_sitk(patient_id, img_sitk, qc_curated_dir_path)

  # if the segmask *is* available
  if has_manual_seg:
    nrrd_reader.SetFileName(patients_data[patient_id][1])
    msk_sitk = nrrd_reader.Execute()
    
    # preprocess the segmasks according to the CT preprocessing parameters
    msk_sitk, curated_size = resample_sitk(img_sitk = msk_sitk,
                                           method = sitk.sitkNearestNeighbor,
                                           curated_spacing = curated_spacing)
    
    msk_sitk, msk_exp_up, msk_exp_dn = expand_sitk(img_sitk = msk_sitk,
                                                   curated_size = curated_size,
                                                   pad_value = 0,
                                                   new_size_up = img_exp_up,
                                                   new_size_down = img_exp_dn)
    
    msk_sitk, new_size_up, new_size_down = crop_sitk(img_sitk = msk_sitk,
                                                     curated_size = curated_size,
                                                     new_size_up = img_crp_up,
                                                     new_size_down = img_crp_dn)
    
    # if the check on the CT and the mask fails, return
    if not check_mask(patient_id, img_sitk, msk_sitk):
      return

    # save preprocessed segmask volume
    nrrd_writer.SetFileName(os.path.join(curated_dir_path, patient_id + '_msk.nrrd'))
    nrrd_writer.SetUseCompression(True)
    nrrd_writer.Execute(msk_sitk)

    # if export_png is set to True when the mask is available, export the quality control png with
    # the segmentation mask overlayed
    if export_png:
      plot_sitk_msk(patient_id, img_sitk, msk_sitk, qc_curated_dir_path)


## ----------------------------------------
## ----------------------------------------

def export_data(raw_data_dir_path, curated_dir_path, qc_curated_dir_path,
                curated_size, curated_spacing, num_cores, export_png, has_manual_seg):

  """
  
  @params:
  
    raw_data_dir_path   - required : input data directory (must contain a folder for each patient)
    curated_dir_path    - required : output directory for the NRRD file(s) (CT and segmask)
    qc_curated_dir_path - required : output directory for the png file
    curated_size        - required : list containing the desired size, in voxels, of the output data
    curated_spacing     - required : list containing the desired spacing, in mm, of the output data
    num_cores           - required : number of cores to use for the multiprocessing 
    export_png          - required : whether to export the quality control png or not
    has_manual_seg      - required : whether a manual segmentation for the volume is available or not

  """
  
  patients_data = dict()
  patient_folders = [x for x in glob(raw_data_dir_path + '/*') if os.path.isdir(x)]
  
  # for every patient in the data folder, populate the dict that will be used to preprocess
  # the data potentially in parallel
  for patient_dir in patient_folders:
    patient_id = os.path.basename(patient_dir)
    img_file = os.path.join(patient_dir, 'img.nrrd')
    msk_file = os.path.join(patient_dir, 'msk.nrrd')
    patients_data[patient_id] = [img_file, msk_file]
  
  print "Data preprocessing:"
  print 'Found', len(patients_data), 'patients under "%s"'%(raw_data_dir_path)

  # if single core, then run core as one would normally do with a function
  if num_cores == 1:
    for patient_id in patients_data:

      run_core(curated_dir_path = curated_dir_path,
               qc_curated_dir_path = qc_curated_dir_path,
               export_png = export_png,
               has_manual_seg = has_manual_seg,
               curated_size = curated_size,
               curated_spacing = curated_spacing,
               patients_data = patients_data,
               patient_id = patient_id)
      
  # else, run the preprocessing in parallel
  elif num_cores > 0:
    pool = Pool(processes = num_cores)
    pool.map(partial(run_core, curated_dir_path, qc_curated_dir_path, export_png, has_manual_seg, curated_size, curated_spacing,
                     patients_data), patients_data.keys())
    pool.close()
    pool.join()
  else:
    print 'Wrong number of CPU cores specified in the config file.'
