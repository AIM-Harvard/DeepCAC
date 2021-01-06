"""
  ----------------------------------------
    CACSeg - run DeepCAC pipeline step3
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 2.7.17
  ----------------------------------------
  
  Deep-learning-based CAC segmentation in
  Chest CT scans - all param.s are parsed
  from a config file stored under "/config"
  
"""

import os
import yaml
import argparse
import matplotlib      
matplotlib.use('Agg')

from step3_cacseg import dilate_segmasks, crop_data, run_inference

## ----------------------------------------

base_conf_file_path = 'config/'
conf_file_list = [f for f in os.listdir(base_conf_file_path) if f.split('.')[-1] == 'yaml']

parser = argparse.ArgumentParser(description = 'Run pipeline step 3 - CAC segmentation.')

parser.add_argument('--conf',
                    required = False,
                    help = 'Specify the YAML configuration file containing the run details.' \
                            + 'Defaults to "cac_segmentation.yaml"',
                    choices = conf_file_list,
                    default = "step3_cac_segmentation.yaml",
                   )


args = parser.parse_args()

conf_file_path = os.path.join(base_conf_file_path, args.conf)

with open(conf_file_path) as f:
  yaml_conf = yaml.load(f, Loader = yaml.FullLoader)


# input-output
data_folder_path = os.path.normpath(yaml_conf["io"]["path_to_data_folder"])

heartloc_data_folder_name = yaml_conf["io"]["heartloc_data_folder_name"]
heartseg_data_folder_name = yaml_conf["io"]["heartseg_data_folder_name"]
cacs_data_folder_name = yaml_conf["io"]["cacs_data_folder_name"]

curated_data_folder_name = yaml_conf["io"]["curated_data_folder_name"]
step2_inferred_data_folder_name = yaml_conf["io"]["step2_inferred_data_folder_name"]

dilated_data_folder_name = yaml_conf["io"]["dilated_data_folder_name"]
cropped_data_folder_name = yaml_conf["io"]["cropped_data_folder_name"]
qc_cropped_data_folder_name = yaml_conf["io"]["qc_cropped_data_folder_name"]

model_weights_folder_name = yaml_conf["io"]["model_weights_folder_name"]
model_output_folder_name = yaml_conf["io"]["model_output_folder_name"]

# preprocessing and inference parameters
has_manual_seg = yaml_conf["processing"]["has_manual_seg"]
export_png = yaml_conf["processing"]["export_png"]
export_cac_slices_png = yaml_conf["processing"]["export_cac_slices_png"]
num_cores = yaml_conf["processing"]["num_cores"]
use_inferred_masks = yaml_conf["processing"]["use_inferred_masks"]
patch_size = yaml_conf["processing"]["patch_size"]

# model config
weights_file_name = yaml_conf["model"]["weights_file_name"]

if has_manual_seg:
  # if manual segmentation masks are available and "use_inferred_masks" is set to False
  # (in the config file), use the manual segmentation masks to compute the localization 
  run = "Test" if use_inferred_masks else "Train"
else:
  
  # signal the user if "use_inferred_masks" if run is forced to "Test"
  if not use_inferred_masks:
    print "Manual segmentation masks not provided. Forcing localization with the inferred masks."
  
  # if manual segmentation masks are not available, force "use_inferred_masks" to True
  run = "Test"

## ----------------------------------------

# set paths: step1, step2 and step3
heartloc_data_path = os.path.join(data_folder_path, heartloc_data_folder_name)
heartseg_data_path = os.path.join(data_folder_path, heartseg_data_folder_name)
cacs_data_path = os.path.join(data_folder_path, cacs_data_folder_name)

# set paths: results from step 1 - data preprocessing
curated_dir_path = os.path.join(heartloc_data_path, curated_data_folder_name) 

# set paths: results from step 2 - heart segmentation
step2_inferred_dir_path = os.path.join(heartseg_data_path, step2_inferred_data_folder_name)

dilated_dir_path = os.path.join(cacs_data_path, dilated_data_folder_name)
cropped_dir_name = os.path.join(cacs_data_path, cropped_data_folder_name)
qc_cropped_dir_name = os.path.join(cacs_data_path, qc_cropped_data_folder_name)

# set paths: model processing
model_weights_dir_path = os.path.join(cacs_data_path, model_weights_folder_name)
model_output_dir_path = os.path.join(cacs_data_path, model_output_folder_name)


# create the subfolders where the results are going to be stored
if not os.path.exists(dilated_dir_path): os.mkdir(dilated_dir_path)
if not os.path.exists(cropped_dir_name): os.mkdir(cropped_dir_name)
if not os.path.exists(qc_cropped_dir_name): os.mkdir(qc_cropped_dir_name)

# assert the curated data folder exists and it is non empty
assert os.path.exists(curated_dir_path)
assert len(os.listdir(curated_dir_path))

# assert the inference data folder exists and it is non empty
assert os.path.exists(step2_inferred_dir_path)
assert len(os.listdir(step2_inferred_dir_path))

# assert the weights folder exists and the weights file is found
weights_file = os.path.join(model_weights_dir_path, weights_file_name)
assert os.path.exists(weights_file)

if not os.path.exists(model_output_dir_path): os.mkdir(model_output_dir_path)

## ----------------------------------------

# run the CAC segmentation pipeline
print "\n--- STEP 3 - CAC SEGMENTATION ---\n"

# 
dilate_segmasks.dilate_segmasks(pred_dir = step2_inferred_dir_path,
                                output_dir = dilated_dir_path,
                                num_cores = num_cores)

#
crop_data.crop_data(raw_input = curated_dir_path,
                    prd_input = dilated_dir_path,
                    data_output = cropped_dir_name,
                    png_output = qc_cropped_dir_name,
                    patch_size = patch_size,
                    num_cores = num_cores,
                    has_manual_seg = has_manual_seg,
                    export_png = export_png)

#
run_inference.run_inference(data_dir = cropped_dir_name,
                            model_weights_dir_path = model_weights_dir_path,
                            weights_file_name = weights_file_name,
                            output_dir = model_output_dir_path,
                            export_cac_slices_png = export_cac_slices_png, 
                            has_manual_seg = has_manual_seg)
