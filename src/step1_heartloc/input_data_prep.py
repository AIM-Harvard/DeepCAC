"""
  ----------------------------------------
     HeartLoc - DeepCAC pipeline step1
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 2.7.17
  ----------------------------------------
  
"""

import os
import sys
import glob
import tables
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage


def get_files(input_dir, create_test_set = None, has_manual_seg = False, train_split = 0.6):
  
  """
  Given a location, parse all the pertinent data files and (potentially) return a train/tune/test split.

  @params:
    input_dir       - required : path to the location where the input data are stored saved
    create_test_set - optional : whether or not to split the input data into training-tuning-test set
    has_manual_seg  - optional : whether a manual segmentation for the volume is available or not
    train_split     - optional : portion of data to be used exclusively for training (defaults to 0.6).
                                 Furthermore, by default the remaining data are split into two halves
                                 (for tuning and testing, respectively) if create_test_set is set to "split".

  """
  
  files = glob.glob(input_dir + '/*')

  patients = list()
  
  # populate
  for file in files:
    
    # if "img" is found in the file name
    if 'img' in file:
      img_file = file
      
      # the path to the mask file (if this exists)
      # should be identical to the image one except for "msk" instead of "img"
      msk_file = img_file.replace('img', 'msk')
      
      # if the image file exists
      if os.path.exists(img_file):
        
        # if the mask should be found (has_manual_seg = True) but it is not, skip the patient
        if has_manual_seg and not os.path.exists(msk_file):
          continue
        
        # append to the patient list the two-elements list with the img/msk file paths
        patients.append([img_file, msk_file])
  
  print 'Found', len(patients), 'patients under "%s" folder.'%(input_dir)

  # train/tune/test split (indices)
  train_split_idx = int(len(patients) * train_split)
  
  # tuning and test set are equally sized by default
  tunetest_split_idx = int((len(patients) - train_split_idx) / 2)
  
  # compute the splits as function of the "create_test_set" argument (case insensitive match)
  if create_test_set.lower() == 'split':
    return patients[0:train_split_idx], \
           patients[train_split_idx:tunetest_split_idx + train_split_idx], \
           patients[tunetest_split_idx + train_split_idx:]
  
  elif create_test_set.lower() == 'none':
    return patients[0:train_split_idx + tunetest_split_idx], \
           patients[tunetest_split_idx + train_split_idx:]
  
  elif create_test_set.lower() == 'all':
    return patients
  
  else:
    print 'Error - "create_test_set" argument should either be "Split", "All" or "None".'


## ----------------------------------------
## ----------------------------------------

def write_data_file(data_dir, data_file, file_list, cube_length, input_spacing, has_manual_seg, fill_mask_holes):

  """
  Description here

  @params:
    data_dir        - required : 
    data_file       - required : 
    file_list       - required : 
    cube_length     - required :
    input_spacing   - required :
    has_manual_seg  - required :
    fill_mask_holes - required :

  """
  
  nrrd_reader = sitk.ImageFileReader()

  # path where the *.h5 file storing ... will be saved
  output_file = os.path.join(data_dir, data_file + '.h5')

  # if the file exists, it will be overwritten
  if os.path.exists(output_file):
    os.remove(output_file)
    
  # init the *.h5 file ptr to file/writer
  hdf5_file = tables.open_file(output_file, mode = 'w')

  # init earrays to properly write on HDF5 files using PyTables  
  pat_id_hdf5 = hdf5_file.create_earray(where = hdf5_file.root, 
                                        name = 'ID', 
                                        atom = tables.StringAtom(itemsize = 65), 
                                        shape = (0,))
  
  img_hdf5 = hdf5_file.create_earray(where = hdf5_file.root,
                                     name = 'img',
                                     atom = tables.FloatAtom(),
                                     shape=(0, cube_length, cube_length, cube_length))
  if has_manual_seg:
    msk_hdf5 = hdf5_file.create_earray(where = hdf5_file.root,
                                       name = 'msk',
                                       atom = tables.UIntAtom(),
                                       shape=(0, cube_length, cube_length, cube_length))

  
  for file in file_list:
    img_file = file[0]
    msk_file = file[1]
    
    pat_id = os.path.basename(img_file).replace('_img.nrrd', '')
    print 'Processing patient', pat_id

    # read SITK image 
    nrrd_reader.SetFileName(img_file)
    img_sitk = nrrd_reader.Execute()
    img_cube = sitk.GetArrayFromImage(img_sitk)
    
    # sanity check on size and spacing
    if not img_sitk.GetSize() == (cube_length, cube_length, cube_length):
      print 'Wrong img SIZE for patient:', pat_id
      continue
    if (not np.round(img_sitk.GetSpacing()[0], 2) == np.round(img_sitk.GetSpacing()[1], 2) ==
            np.round(img_sitk.GetSpacing()[2], 2) == input_spacing):
      print 'Wrong img SPACING for patient:', pat_id, img_sitk.GetSpacing()
      continue

    # crop image from 100px to 96px (as 96 = 12*2^3 is CNN friendly)
    input_length = len(img_cube)
    offset = int((input_length - cube_length) / 2)
    
    img_cropped = np.copy(img_cube[offset:offset + cube_length,
                                   offset:offset + cube_length,
                                   offset:offset + cube_length])
    
    # normalise intensity values between 0 (+ eps) and 1
    img_cropped = ((np.clip(img_cropped, -1024.0, 3071.0)) - 1023.5) / 2047.5
    
    # store ID as a node in the HDF5 vector
    pat_id_hdf5.append(np.array([pat_id], dtype='S65'))
    
    # store image as a node in the HDF5 vector
    img_hdf5.append(img_cropped[np.newaxis, ...])

    # if a (manual) segmentation mask is available   
    if has_manual_seg:
      nrrd_reader.SetFileName(msk_file)
      msk_sitk = nrrd_reader.Execute()
      msk_cube = sitk.GetArrayFromImage(msk_sitk)
      
      # sanity check on size and spacing
      if not img_sitk.GetSize() == msk_sitk.GetSize():
        print 'Wrong msk SIZE for patient:', pat_id
        continue
      if (not np.round(msk_sitk.GetSpacing()[0], 2) == np.round(msk_sitk.GetSpacing()[1], 2) ==
              np.round(msk_sitk.GetSpacing()[2], 2) == input_spacing):
        print 'Wrong msk SPACING for patient:', pat_id, msk_sitk.GetSpacing()
        continue
      
      # crop mask from 100px to 96px (as 96 = 12*2^3 is CNN friendly)
      msk_cropped = np.copy(msk_cube[offset:offset + cube_length,
                                     offset:offset + cube_length,
                                     offset:offset + cube_length])

      # FIXME: this will raise an error if the mask is not properly formatted (e.g., has only one label!)
      # transform into binary mask
      msk_cropped[msk_cropped == 2] = 0
      msk_cropped[msk_cropped > 0] = 1
      
      # fill holes in the mask if "fill_mask_holes" is set to true
      if fill_mask_holes:
        for slice_idx in range(len(msk_cropped)):
          msk_cropped[slice_idx] = ndimage.binary_fill_holes(msk_cropped[slice_idx])

      # store mask as a node in the HDF5 vector
      msk_hdf5.append(msk_cropped[np.newaxis, ...])
  
  hdf5_file.close()


## ----------------------------------------
## ----------------------------------------

def input_data_prep(resampled_dir_path, model_input_dir_path, create_test_set, crop_size, new_spacing, has_manual_seg, fill_mask_holes):
  
  print "\nInput data preparation:"
  
  if create_test_set == 'Split':
    print 'Loading input data from "%s"'%(resampled_dir_path)
    
    trainFiles, testFiles, valFiles = get_files(resampled_dir_path, create_test_set, has_manual_seg)
    
    print 'Writing data file... '
    write_data_file(model_input_dir_path, "step1_training_data", trainFiles, crop_size, new_spacing, has_manual_seg, fill_mask_holes)
    write_data_file(model_input_dir_path, "step1_val_data", valFiles, crop_size, new_spacing, has_manual_seg, fill_mask_holes)
    write_data_file(model_input_dir_path, "step1_test_data", testFiles, crop_size, new_spacing, has_manual_seg, fill_mask_holes)
  
  elif create_test_set == 'None':
    print 'Loading input data from "%s"'%(resampled_dir_path)
    
    trainFiles, valFiles = get_files(resampled_dir_path, create_test_set, has_manual_seg)
    
    print 'Writing data file... '
    write_data_file(model_input_dir_path, "step1_training_data", trainFiles, crop_size, new_spacing, has_manual_seg, fill_mask_holes)
    write_data_file(model_input_dir_path, "step1_val_data", valFiles, crop_size, new_spacing, has_manual_seg, fill_mask_holes)
  
  elif create_test_set == 'All':
    print 'Loading input data from "%s"'%(resampled_dir_path)
    
    testFiles = get_files(resampled_dir_path, create_test_set, has_manual_seg)
    
    print 'Writing data file... '
    write_data_file(model_input_dir_path, "step1_test_data", testFiles, crop_size, new_spacing, has_manual_seg, fill_mask_holes)
  
  else:
    print 'Error - wrong setting for "createTestSet"'
