
"""
  ----------------------------------------
     HeartLoc - DeepCAC pipeline step1
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 2.7.17
  ----------------------------------------

  After the data (CT-mask pair, or just CT) is processed by the first script,
  export downsampled versions to be used for heart-localisation purposes.
  During this downsampling step, resample and crop/pad images - log all the
  information needed for upsampling (and thus obtain a rough segmentation that
  will be used for the localisation).
  
"""


import os
import sys
import numpy as np
import multiprocessing
import cPickle as pickle
import SimpleITK as sitk

from glob import glob
from functools import partial


def resample_sitk(data_sitk, new_spacing, method):

  """
  Downsample volumes exploiting Simple ITK. 

  @params:
    data_sitk   - required : SimpleITK image, resulting from sitk.ImageFileReader().Execute()
    new_spacing - required : desired spacing (equal for all the axes), in mm, of the output data
    method      - required : SimpleITK interpolation method (e.g., sitk.sitkLinear)

  FIXME: change this into something like downsample_sitk (also, data_sitk to img_sitk for homog.)
  (as this function is especially used for downsampling, right?)
  Alternatively, exploit "resample_sitk" in exportData.py, maybe creating a lib folder and making
  some of the modules reusable (AIM would greatly benefit by this anyway)
  """

  # get size, in voxels, and spacing, in mm, of the SITK image to downsample
  orig_size = data_sitk.GetSize()
  orig_spacing = data_sitk.GetSpacing()

  # compute the new size, in voxels
  new_size = int(orig_size[0]*orig_spacing[0]/new_spacing)

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

  res_filter = sitk.ResampleImageFilter()
  img_sitk = res_filter.Execute(data_sitk, [new_size, new_size, new_size], sitk.Transform(), method,
                                data_sitk.GetOrigin(), [new_spacing, new_spacing, new_spacing],
                                data_sitk.GetDirection(), 0, data_sitk.GetPixelIDValue())
  
  return img_sitk, orig_size, orig_spacing


## ----------------------------------------
## ----------------------------------------

def resize_sitk(data_sitk, crop_size, pad_value = -1024):

  """
  Volumes cropping and in-plane padding exploiting Simple ITK. The resulting image will be sized
  [crop_size, crop_size, crop_size]. Furthermore, the correct physical location of the image is maintained.

  @params:
    data_sitk - required : SimpleITK image, resulting from sitk.ImageFileReader().Execute()
    crop_size - required : desired size (equal for all the axes), in voxels, of the output data
    pad_value - required : constant value used to pad the image (defaults to the HU of air, -1024)

    # FIXME: change "crop_size" to something more significant?
  """

  ## ----------------------------------------
  # crop volumes which are bigger than "crop_size" (all the axes)

  # size in voxels of the volume to crop
  old_size = data_sitk.GetSize()
  
  # compute the difference, in voxels, between the size of the volume to crop and crop_size (output)
  size_diff = [crop_size - old_size[0], crop_size - old_size[1], crop_size - old_size[2]]

  # if old_size < crop_size, then the second argument of max is negative and the max is zero,
  # so nothing gets cropped (image lower sides, so half of the crop on this side)
  new_size_down = [max(0, int((old_size[0] - crop_size) / 2)),
                   max(0, int((old_size[1] - crop_size) / 2)),
                   max(0, int((old_size[2] - crop_size) / 2))]

  # same logic, but for image upper sides
  new_size_up = [max(0, old_size[0] - crop_size - new_size_down[0]),
                 max(0, old_size[1] - crop_size - new_size_down[1]),
                 max(0, old_size[2] - crop_size - new_size_down[2])]

  crop_filter = sitk.CropImageFilter()

  crop_filter.SetUpperBoundaryCropSize(new_size_up)
  crop_filter.SetLowerBoundaryCropSize(new_size_down)
  
  data_sitk = crop_filter.Execute(data_sitk)

  ## ----------------------------------------
  # pad volumes which are smaller than "crop_size" (all the axes)
  
  # size in voxels of the volume to crop (after cropping)
  old_size = data_sitk.GetSize()

  # if old_size > crop_size, then the second argument of max is negative and the max is zero,
  # so nothing gets cropped (image lower sides, so half of the crop on this side)
  new_size_down = [max(0, int((crop_size - old_size[0]) / 2)),
                   max(0, int((crop_size - old_size[1]) / 2)),
                   max(0, int((crop_size - old_size[2]) / 2))]
  
  # same logic, but for image upper sides
  new_size_up = [max(0, crop_size - old_size[0] - new_size_down[0]),
                 max(0, crop_size - old_size[1] - new_size_down[1]),
                 max(0, crop_size - old_size[2] - new_size_down[2])]

  pad_filter = sitk.ConstantPadImageFilter()

  # value to pad with, defaults to the HU of air
  pad_filter.SetConstant(pad_value)
  pad_filter.SetPadUpperBound(new_size_up)
  pad_filter.SetPadLowerBound(new_size_down)

  data_sitk = pad_filter.Execute(data_sitk)

  final_size = data_sitk.GetSize()

  # spacing should not be changed, but useful return for sanity check anyways
  final_spacing = data_sitk.GetSpacing()

  return data_sitk, final_size, final_spacing, size_diff


## ----------------------------------------
## ----------------------------------------

def save_nrrd(data_sitk, outfile_path):

  """
  Save a SITK volume as a *.nrrd file.

  @params:
    data_sitk    - required : SimpleITK image, resulting from sitk.ImageFileReader().Execute()
                              (and various other SimpleITK operations)
    outfile_path - required : path to the location where the output file should be saved
                              (including the file name and extension)
  
  """

  nrrd_writer = sitk.ImageFileWriter()
  nrrd_writer.SetFileName(outfile_path)
  nrrd_writer.SetUseCompression(True)
  nrrd_writer.Execute(data_sitk)


## ----------------------------------------
## ----------------------------------------

def run_core(resampled_dir_path, crop_size, new_spacing, has_manual_seg, result_dict, img_file):

  """
   Downsampling core function to be run (potentially) with multiprocessing.

  @params:
    resampled_dir_path - required : output directory where to store the file(s) (downsampled CT and segmask)
    crop_size          - required : desired size (equal for all the axes), in voxels, of the output data
    new_spacing        - required : desired spacing (equal for all the axes), in mm, of the output data
    has_manual_seg     - required : whether a manual segmentation for the volume is available or not
    result_dict        - required : (pointer to) dictionary where to save (pickable) file storing image properties
    img_file           - required : path to the location where the subject .*nrrd file to be processed
                                    is stored (including the file name and extension)
  
  """

  nrrd_reader = sitk.ImageFileReader()
  patient_id = os.path.basename(img_file).replace('_img.nrrd', '')
  img_out_file = os.path.join(resampled_dir_path, patient_id + '_img.nrrd')

  print 'Processing patient', patient_id

  # read the patient image data
  try:
    nrrd_reader.SetFileName(img_file)
    img_sitk = nrrd_reader.Execute()
  except:
    print 'EXCEPTION: Unable to read NRRD image file for patient', patient_id
    return

  # downsample CT data
  img_sitk, orig_size_img, orig_spacing_img = resample_sitk(data_sitk = img_sitk, 
                                                            new_spacing = new_spacing,
                                                            method = sitk.sitkLinear)

  # resize (crop/pad) the resulting image to [crop_size, crop_size, crop_size]
  img_sitk, final_size_img, final_spacing_img, size_diff_img = resize_sitk(data_sitk = img_sitk,
                                                                           crop_size = crop_size,
                                                                           pad_value = -1024)

  # sanity checks on the size and spacing of the resulting image
  if not (crop_size, crop_size, crop_size) == final_size_img:
    print 'Wrong final IMG size', patient_id, final_size_img
  if not (new_spacing, new_spacing, new_spacing) == final_spacing_img:
    print 'Wrong final IMG spacing', patient_id, final_spacing_img

  # if everything is all right, save the downsampled and resized CT image as *.nrrd
  save_nrrd(data_sitk = img_sitk, outfile_path = img_out_file)

  msk_file = ''

  # if a manual segmentation is associated to the patient CT
  if has_manual_seg:
    # path to the mask file
    msk_file = img_file.replace('img', 'msk')
    
    # if the mask file exists
    if os.path.exists(msk_file):
      
      # sanity check - read the *.nrrd file for the mask 
      try:
        nrrd_reader.SetFileName(msk_file)
        msk_sitk = nrrd_reader.Execute()
      except Exception as e:
        print 'EXCEPTION: Unable to read NRRD mask file for patient', patient_id, e
        return

      # downsample segmask data
      msk_sitk, orig_size_mask, orig_spacing_mask = resample_sitk(data_sitk = msk_sitk,
                                                                  new_spacing = new_spacing,
                                                                  method = sitk.sitkNearestNeighbor)
      
      # resize (crop/pad) the resulting image to [crop_size, crop_size, crop_size]
      msk_sitk, final_size_mask, final_spacing_mask, size_dif_mask = resize_sitk(data_sitk = msk_sitk,
                                                                                 crop_size = crop_size,
                                                                                 pad_value = 0)

      # save such mask file
      msk_out_file = os.path.join(resampled_dir_path, patient_id + '_msk.nrrd')
      save_nrrd(data_sitk = msk_sitk, outfile_path = msk_out_file)

      # FIXME: try-except instead of returns?
      # sanity checks on the size and spacing of the original mask
      if not tuple(np.round(orig_size_img, 1)) == tuple(np.round(orig_size_mask, 1)):
        print 'Wrong original MSK size', patient_id, tuple(np.round(orig_size_img, 1)), tuple(np.round(orig_size_mask, 1))
        return
      if not tuple(np.round(orig_spacing_img, 1)) == tuple(np.round(orig_spacing_mask, 1)):
        print 'Wrong original MSK spacing', patient_id, tuple(np.round(orig_spacing_img, 1)), tuple(np.round(orig_spacing_mask, 1))
        return

      # sanity checks on the size and spacing of the resulting mask
      if not tuple(np.round(final_size_img, 1)) == tuple(np.round(final_size_mask, 1)):
        print 'Wrong final MSK size', patient_id, tuple(np.round(final_size_img, 1)), tuple(np.round(final_size_mask, 1))
        return
      if not tuple(np.round(final_spacing_mask, 1)) == tuple(np.round(final_spacing_mask, 1)):
        print 'Wrong final MSK spacing', patient_id, tuple(np.round(final_spacing_img, 1)), tuple(np.round(final_spacing_mask, 1))
        return
    # if the mask file does not exist (has_manual_seg == True, so it should), return
    else:
      print('WARNING: MSK file not found. Skipping', patient_id)
      return

  # populate the patient_id entry of result_dict (passed as a reference) with:
  # - path to the location where the subject CT .*nrrd file to be processed is stored
  # - path to the location where the subject segmask .*nrrd file to be processed is stored
  # - size and spacing of the CT image (before downsampling) (list)
  # - size and spacing of the CT image (after downsampling and cropping/padding) (list)
  # - difference in size between the two images (list)
  result_dict[patient_id] = [img_file, msk_file,
                             orig_size_img, orig_spacing_img,
                             final_size_img, final_spacing_img,
                             size_diff_img]


## ----------------------------------------
## ----------------------------------------

def downsample_data(curated_dir_path, resampled_dir_path, model_input_dir_path, 
                    crop_size, new_spacing, has_manual_seg, num_cores):

  """
  FIXME: add description? Recap all the steps. Also, change/shorten some of the argument names.

  @params:
    curated_dir_path     - required : input data directory (must contain a (pair of img-msk) file(s) for each patient)
    resampled_dir_path   - required : output directory where to store the file(s) (downsampled CT and segmask)
    model_input_dir_path - required : output directory where to store the results dictionary
                                      (logs some image properties for the downsampling step, see "run_core")
    crop_size            - required : desired size (equal for all the axes), in voxels, of the output data
    new_spacing          - required : desired spacing (equal for all the axes), in mm, of the output data
    has_manual_seg       - required : whether a manual segmentation for the volume is available or not
    num_cores            - required : number of cores to use for the multiprocessing 

  """
  
  print "\nData downsampling:"
  
  img_files = glob(curated_dir_path + '/*_img.nrrd')
  
  print 'Found', len(img_files), 'patients under "%s" folder.'%(curated_dir_path)

  # if single core, then run core as one would normally do with a function
  if num_cores == 1:
    result_dict = {}
    for img_file in img_files:
      run_core(resampled_dir_path = resampled_dir_path,
               crop_size = crop_size,
               new_spacing = new_spacing,
               has_manual_seg = has_manual_seg,
               result_dict = result_dict,
               img_file = img_file)

  # else, run the preprocessing in parallel
  elif num_cores > 0:
    with multiprocessing.Manager() as manager:
      result_dict = manager.dict()
      pool = multiprocessing.Pool(processes = num_cores)
      pool.map(partial(run_core, resampled_dir_path, crop_size, new_spacing, has_manual_seg, result_dict), img_files)
      pool.close()
      pool.join()
      result_dict = dict(result_dict)
  else:
    print 'Wrong number of CPU cores specified in the config file.'
    sys.exit()

  # save pkl file for image properties
  results_file_name = os.path.join(model_input_dir_path, 'step1_downsample_results.pkl')
  
  print 'Saving results dictionary...'
  with open(results_file_name, 'wb') as results_file:
    pickle.dump(result_dict, results_file, pickle.HIGHEST_PROTOCOL)
