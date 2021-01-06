"""
  ----------------------------------------
    HeartSeg - run DeepCAC pipeline step2
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 2.7.17
  ----------------------------------------
  
  Deep-learning-based heart segmentation in
  Chest CT scans - all param.s are parsed
  from a config file stored under "/config"
  
"""

import os
import yaml
import argparse
import matplotlib      
matplotlib.use('Agg')

from step2_heartseg import compute_bbox, crop_data, input_data_prep, run_inference, upsample_results, compute_metrics

## ----------------------------------------

base_conf_file_path = 'config/'
conf_file_list = [f for f in os.listdir(base_conf_file_path) if f.split('.')[-1] == 'yaml']

parser = argparse.ArgumentParser(description = 'Run pipeline step 2 - heart segmentation.')

parser.add_argument('--conf',
                    required = False,
                    help = 'Specify the YAML configuration file containing the run details.' \
                            + 'Defaults to "heart_segmentation.yaml"',
                    choices = conf_file_list,
                    default = "step2_heart_segmentation.yaml",
                   )


args = parser.parse_args()

conf_file_path = os.path.join(base_conf_file_path, args.conf)

with open(conf_file_path) as f:
  yaml_conf = yaml.load(f, Loader = yaml.FullLoader)


# input-output
data_folder_path = os.path.normpath(yaml_conf["io"]["path_to_data_folder"])

heartloc_data_folder_name = yaml_conf["io"]["heartloc_data_folder_name"]
heartseg_data_folder_name = yaml_conf["io"]["heartseg_data_folder_name"]

curated_data_folder_name = yaml_conf["io"]["curated_data_folder_name"]
step1_inferred_data_folder_name = yaml_conf["io"]["step1_inferred_data_folder_name"]

bbox_folder_name = yaml_conf["io"]["bbox_folder_name"]
cropped_data_folder_name = yaml_conf["io"]["cropped_data_folder_name"]
model_input_folder_name = yaml_conf["io"]["model_input_folder_name"]
model_weights_folder_name = yaml_conf["io"]["model_weights_folder_name"]
model_output_folder_name = yaml_conf["io"]["model_output_folder_name"]
upsampled_data_folder_name = yaml_conf["io"]["upsampled_data_folder_name"]
seg_metrics_folder_name = yaml_conf["io"]["seg_metrics_folder_name"]

# preprocessing and inference parameters
has_manual_seg = yaml_conf["processing"]["has_manual_seg"]
fill_mask_holes = yaml_conf["processing"]["fill_mask_holes"]
export_png = yaml_conf["processing"]["export_png"]
num_cores = yaml_conf["processing"]["num_cores"]
use_inferred_masks = yaml_conf["processing"]["use_inferred_masks"]
curated_size = yaml_conf["processing"]["curated_size"]
curated_spacing = yaml_conf["processing"]["curated_spacing"]
inter_size = yaml_conf["processing"]["inter_size"]
training_size = yaml_conf["processing"]["training_size"]
final_size = yaml_conf["processing"]["final_size"]
final_spacing = yaml_conf["processing"]["final_spacing"]

# model config
weights_file_name = yaml_conf["model"]["weights_file_name"]
down_steps = yaml_conf["model"]["down_steps"]


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

# set paths: step1 and step2
heartloc_data_path = os.path.join(data_folder_path, heartloc_data_folder_name)
heartseg_data_path = os.path.join(data_folder_path, heartseg_data_folder_name)

# set paths: results from step 1 - heart localisation
curated_dir_path = os.path.join(heartloc_data_path, curated_data_folder_name)
step1_inferred_dir_path = os.path.join(heartloc_data_path, step1_inferred_data_folder_name)

bbox_dir_path = os.path.join(heartseg_data_path, bbox_folder_name)
cropped_dir_name = os.path.join(heartseg_data_path, cropped_data_folder_name)

# set paths: model processing
model_input_dir_path = os.path.join(heartseg_data_path, model_input_folder_name)
model_weights_dir_path = os.path.join(heartseg_data_path, model_weights_folder_name)
model_output_dir_path = os.path.join(heartseg_data_path, model_output_folder_name)

# set paths: final location where the inferred masks (NRRD) and the metrics,
# computed if the manual segmentation masks are available, will be stored
model_output_nrrd_dir_path = os.path.join(heartseg_data_path, upsampled_data_folder_name)
result_metrics_dir_path = os.path.join(heartseg_data_path, seg_metrics_folder_name)


# create the subfolders where the results are going to be stored
if not os.path.exists(bbox_dir_path): os.mkdir(bbox_dir_path)
if not os.path.exists(cropped_dir_name): os.mkdir(cropped_dir_name)
if not os.path.exists(model_input_dir_path): os.mkdir(model_input_dir_path)

# assert the curated data folder exists and it is non empty
assert os.path.exists(curated_dir_path)
assert len(os.listdir(curated_dir_path))

# assert the inference data folder exists and it is non empty
assert os.path.exists(step1_inferred_dir_path)
assert len(os.listdir(step1_inferred_dir_path))


# assert the weights folder exists and the weights file is found
weights_file = os.path.join(model_weights_dir_path, weights_file_name)
assert os.path.exists(weights_file)

if not os.path.exists(model_output_dir_path): os.mkdir(model_output_dir_path)
if not os.path.exists(model_output_nrrd_dir_path): os.mkdir(model_output_nrrd_dir_path)
if not os.path.exists(result_metrics_dir_path): os.mkdir(result_metrics_dir_path)

## ----------------------------------------

# run the segmentation pipeline
print "\n--- STEP 2 - HEART SEGMENTATION ---\n"

# 
compute_bbox.compute_bbox(cur_dir = curated_dir_path,
                          pred_dir = step1_inferred_dir_path,
                          output_dir = bbox_dir_path,
                          num_cores = num_cores,
                          has_manual_seg = has_manual_seg,
                          run = run)

#
crop_data.crop_data(bb_calc_dir = bbox_dir_path,
                    output_dir = cropped_dir_name, 
                    network_dir = model_input_dir_path,
                    inter_size = inter_size,
                    final_size = final_size,
                    final_spacing = final_spacing,
                    num_cores = num_cores)

#
input_data_prep.input_data_prep(input_dir = cropped_dir_name,
                                output_dir = model_input_dir_path,
                                run = run,
                                fill_holes = fill_mask_holes,
                                final_size = final_size)

#
run_inference.run_inference(model_weights_dir_path = model_weights_dir_path,
                            data_dir = model_input_dir_path,
                            output_dir = model_output_dir_path,
                            weights_file_name = weights_file_name,
                            export_png = export_png,
                            final_size = final_size,
                            training_size = training_size,
                            down_steps = down_steps)

#
upsample_results.upsample_results(cur_input = curated_dir_path,
                                  crop_input = cropped_dir_name,
                                  network_dir = model_input_dir_path,
                                  test_dir = model_output_dir_path,
                                  output_dir = model_output_nrrd_dir_path,
                                  inter_size = inter_size,
                                  num_cores = num_cores)

#
if has_manual_seg == True:
  compute_metrics.compute_metrics(cur_dir = curated_dir_path,
                                  pred_dir = model_output_nrrd_dir_path,
                                  output_dir = result_metrics_dir_path,
                                  raw_spacing = curated_spacing,
                                  num_cores = num_cores,
                                  mask = has_manual_seg)
else:
  pass
