"""
  ----------------------------------------
   CACScoring - run DeepCAC pipeline step4
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 2.7.17
  ----------------------------------------
  
  Agatston CAC scoring from Chest CT scans
  All param.s are parsed from a config file
  stored under "/config"
  
"""

import os
import csv
import yaml
import argparse
import numpy as np
import SimpleITK as sitk

from glob import glob
from functools import partial
from scipy.ndimage import measurements
from multiprocessing import Pool, Manager

## ----------------------------------------

base_conf_file_path = 'config/'
conf_file_list = [f for f in os.listdir(base_conf_file_path) if f.split('.')[-1] == 'yaml']

parser = argparse.ArgumentParser(description = 'Run pipeline step 4 - CAC scoring.')

parser.add_argument('--conf',
                    required = False,
                    help = 'Specify the YAML configuration file containing the run details.' \
                            + 'Defaults to "cac_scoring.yaml"',
                    choices = conf_file_list,
                    default = "step4_cac_scoring.yaml",
                   )


args = parser.parse_args()

conf_file_path = os.path.join(base_conf_file_path, args.conf)

with open(conf_file_path) as f:
  yaml_conf = yaml.load(f, Loader = yaml.FullLoader)


# input-output
data_folder_path = os.path.normpath(yaml_conf["io"]["path_to_data_folder"])

heartloc_data_folder_name = yaml_conf["io"]["heartloc_data_folder_name"]
heartseg_data_folder_name = yaml_conf["io"]["heartseg_data_folder_name"]
cacseg_data_folder_name = yaml_conf["io"]["cacseg_data_folder_name"]

curated_data_folder_name = yaml_conf["io"]["curated_data_folder_name"]
step3_inferred_data_folder_name = yaml_conf["io"]["step3_inferred_data_folder_name"]

cropped_data_folder_name = yaml_conf["io"]["cropped_data_folder_name"]

cac_score_folder_name = yaml_conf["io"]["cac_score_folder_name"]

# preprocessing and inference parameters
has_manual_seg = yaml_conf["processing"]["has_manual_seg"]
num_cores = yaml_conf["processing"]["num_cores"]

## ----------------------------------------

# set paths: step1, step2 and step3
heartloc_data_path = os.path.join(data_folder_path, heartloc_data_folder_name)
heartseg_data_path = os.path.join(data_folder_path, heartseg_data_folder_name)
cacseg_data_path = os.path.join(data_folder_path, cacseg_data_folder_name)

# set paths: results from step 1 - data preprocessing
curated_dir_path = os.path.join(heartloc_data_path, curated_data_folder_name) 

# set paths: results from step 3 - cac segmentation
cropped_dir_name = os.path.join(cacseg_data_path, cropped_data_folder_name)
step3_inferred_dir_path = os.path.join(cacseg_data_path, step3_inferred_data_folder_name)

# set paths: final location where the results of the step4 (CAC scores) will be stored
cacscore_data_path = os.path.join(data_folder_path, cac_score_folder_name)

if not os.path.exists(cacscore_data_path): os.mkdir(cacscore_data_path)

## ----------------------------------------
## ----------------------------------------

def dice_coef(y_true, y_pred, smooth=1.):
  y_true_f = y_true.flatten()
  y_pred_f = y_pred.flatten()
  return (2. * np.sum(y_true_f * y_pred_f) + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

## ----------------------------------------

def get_ag_class(ag):
  ag = round(ag, 3)
  class_ag = None
  if ag == 0:
    class_ag = 0
  if 0 < ag <= 100:
    class_ag = 1
  if 100 < ag <= 300:
    class_ag = 2
  if ag > 300:
    class_ag = 3
  return class_ag

## ----------------------------------------

def get_object_ag(clc_object, object_volume):
  object_max = np.max(clc_object)
  object_ag = 0
  if 130 <= object_max < 200:
    object_ag = object_volume * 1
  if 200 <= object_max < 300:
    object_ag = object_volume * 2
  if 300 <= object_max < 400:
    object_ag = object_volume * 3
  if object_max >= 400:
    object_ag = object_volume * 4
  return object_ag

## ----------------------------------------

def get_ag(img_cube, msk_cube, nr_con_px, spacing, ag_div):
  ag = 0
  px_volume = round(spacing[0] * spacing[1] * spacing[2] / ag_div, 3)

  # Loop over all slices:
  for slice_nr in range(len(msk_cube)):
    img_slice = img_cube[slice_nr]
    msk_slice = msk_cube[slice_nr]

    # Get all objects in mask
    labeled_mask, num_labels = measurements.label(msk_slice, structure=np.ones((3, 3)))

    # Process each object
    for label_nNr in range(1, num_labels + 1):
      label = np.zeros(msk_slice.shape)
      label[labeled_mask == label_nNr] = 1
      clc_object = img_slice * label

      # 1) Remove small objects
      if np.sum(label) <= nr_con_px:
        continue
      # 2) Calculate volume
      object_volume = np.sum(label) * px_volume
      # 3) Calculate AG for object
      object_ag = round(get_object_ag(clc_object, object_volume), 3)
      # 4) Sum up scores
      ag += object_ag

  return ag

## ----------------------------------------

def run_core(res_dict, raw_dir, crop_dir, mask, ag_div, nr_con_px, msk_thr, prd_file):
  nrrd_reader = sitk.ImageFileReader()
  patient_id = os.path.basename(prd_file).replace('_pred.npy', '')

  print 'Processing patient', patient_id

  img_file = os.path.join(crop_dir, patient_id + '_img_3071.npy')
  if not os.path.exists(img_file) or not os.path.exists(prd_file):
    print 'Error - IMG 3071 not found', patient_id
    print img_file, '\n', prd_file
    return

  nrrd_file = os.path.join(raw_dir, patient_id + '_img.nrrd')
  if not os.path.exists(nrrd_file):
    print 'Error - IMG not found', patient_id, nrrd_file
    return

  img = np.load(img_file)
  prd = np.load(prd_file)

  prd[prd < msk_thr] = 0
  prd[prd > 0] = 1

  nrrd_reader.SetFileName(nrrd_file)
  img_raw_sitk = nrrd_reader.Execute()
  spacing = img_raw_sitk.GetSpacing()

  # Calculate the Agatston score and its risk score
  cac_pred = round(get_ag(img, prd, nr_con_px, spacing, ag_div), 3)
  class_pred = get_ag_class(cac_pred)

  if mask:
    msk_file = os.path.join(crop_dir, patient_id + '_msk.npy')
    if not os.path.exists(msk_file):
      print 'Error - MSK not found', patient_id, msk_file
      return
    msk = np.load(msk_file)
    cac_calc = round(get_ag(img, msk, nr_con_px, spacing, ag_div), 3)
    class_calc = get_ag_class(cac_calc)

    dice = round(np.round(dice_coef(msk, prd, smooth=1.), 3), 3)
  else:
    cac_calc = -1
    class_calc = -1
    dice = -1

  # Save the results to the multiprocessing enabled dictionary
  res_dict[patient_id] = [dice, cac_calc, cac_pred, class_calc, class_pred]

## ----------------------------------------
## ----------------------------------------

if __name__ == "__main__":
 
  print "\n--- STEP 4 - CAC SCORING ---\n"
   
  msk_thr = 0.1
  nr_con_px = 3
  ag_div = 3
  
  step3_inferred_dir_path_npy = os.path.join(step3_inferred_dir_path, 'npy')
  pred_files = glob(step3_inferred_dir_path_npy + '/*.npy')
  print 'Found', len(pred_files), 'patients under "%s"'%(step3_inferred_dir_path_npy)

  if num_cores == 1:
    res_dict = {}
    for pred_file in pred_files:
      run_core(res_dict = res_dict,
               raw_dir = curated_dir_path,
               crop_dir = cropped_dir_name,
               mask = has_manual_seg,
               ag_div = ag_div,
               nr_con_px = nr_con_px,
               msk_thr = msk_thr,
               prd_file = pred_file)
  else:
    with Manager() as manager:
      res_dict = manager.dict()
      pool = Pool(processes = num_cores)
      pool.map(partial(run_core, res_dict, curated_dir_path, cropped_dir_name,
                       has_manual_seg, ag_div, nr_con_px, msk_thr),
               pred_files)
      
      pool.close()
      pool.join()
      res_dict = dict(res_dict)

  print '\nSaving results to csv file under "%s"'%(cacscore_data_path)
  result_file = os.path.join(cacscore_data_path, 'cac_score_results.csv')

  with open(result_file, 'wb') as csv_file:
    file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting = csv.QUOTE_MINIMAL)
    title = ['PatientID', 'Dice', 'CAC_calc', 'CAC_pred', 'Class_calc', 'Class_pred']
    file_writer.writerow(title)

    for patient_id in res_dict:
      row_list = [patient_id]
      row_list += res_dict[patient_id]
      file_writer.writerow(row_list)

  # Calculate mean dice
  mean_dice = sum([res_dict[x][0] for x in res_dict.keys()]) / len(res_dict)
  print '\nMean Dice Score:', round(mean_dice, 2)
