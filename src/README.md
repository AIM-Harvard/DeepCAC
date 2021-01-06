# Source Code

Here follows a brief description of how the source code is organised and what are the different steps of the processing.

Additional information regarding the output of the pipeline (e.g., where the results of a certain step will be exported) are found in the markdown file under the `data` directory of the repository.

## Step 1: Heart Localization

The heart localization step takes care of the following operations:

1. Data preprocessing (`export_data.py`);
2. Data downsampling (for heart localisation purposes, `downsample_data.py`);
3. Prepare the data to be processed in a network-friendly format (`input_data_prep.py`);
4. Run the heart localization Deep Learning model on the downsampled data (`run_inference.py`). The architecture of the Deep Learning model trained for the heart localization can be found in `heartloc_model.py`;
5. Upsample the inferred segmentation masks to the size and spacing of the preprocessed data (`upsample_results.py`).

The heart localization step can be run by executing:

```
python run_step1_heart_localization.py
```
```
python run_step1_heart_localization.py --conf ${NAME_OF_CONF_FILE}.yaml
```

By default, `config/step1_heart_localization.yaml` is loaded. All the tweakable parameters (e.g., path to the base data folder, output folders naming, wheter or not to export png images for quality control purposes) for this first step can be found (and modified!) in such file.

### Brief Description

The data preprocessing step consists of a series of operations necessary to ensure the input data look sufficiently similar to the data the model was trained on. First, each volume is resampled to a common resolution (specified in the configuration file for the given step). Second, a proper padding/cropping is applied to the resampled data (mantaining the physical location of the image). Finally, all the data are checked for errors and exported. Multiprocessing can be enabled setting the corresponding flag in the configuration file.

Together with the data, quality-control images are exported by default (this behaviour can be controlled, again, by modifying the correct flags in the configuration file).

Heart localization is achieved by computing a rough segmentation mask of the organ (from which is easy to compute parameters such as the center of mass). Therefore, a downsampling step is run before inference. After the latter, data upsampling is exploited to make size and spacing of the rough segmentation masks coherent with those of the preprocessed data.

## Step 2: Heart Segmentation

The heart segmentation step takes care of the following operations:

1. Compute the heart bounding box (`compute_bbox.py`);
2. Data cropping (for heart segmentation purposes, `crop_data.py`);
3. Prepare the data to be processed in a network-friendly format (`input_data_prep.py`);
4. Run the heart segmentation Deep Learning model on the cropped data (`run_inference.py`). The architecture of the Deep Learning model trained for the heart localization can be found in `heartseg_model.py`;
5. Upsample the inferred segmentation masks to the size and spacing of the preprocessed data (`upsample_results.py`);
6. If a ground truth for the processed volumes is available, compute some segmentation metrics for quality control (`compute_metrics.py`).

The heart segmentation step can be run by executing:

```
python run_step2_heart_segmentation.py
```
```
python run_step2_heart_segmentation.py --conf ${NAME_OF_CONF_FILE}.yaml
```

By default, `config/step2_heart_segmentation.yaml` is loaded. All the tweakable parameters (e.g., path to the heart localisation data folder, output folders naming, wheter or not to export png images for quality control purposes) for this second step can be found (and modified!) in such file.

### Brief Description

In order to increase as much as possible the number of parameters of the Deep Learning heart segmentation model, and the batch size during training, the CT volume is not processed by the heart segmentation in its entirety. Rather, a crop is computed starting from the rough segmentation inferred in the first step. At first, a bounding box is computed for such segmentation mask. Then, a subvolume of the CT which contains the heart is cropped from each preprocessed volume. Finally, the Deep Learning inference is run and the resulting masks processed such that their spacing, size and position in the patient space is coherent with those of the preprocessed data.

## Step 3: CAC Segmentation

The coronary artery calcium (CAC) segmentation step takes care of the following operations:

1. Segmentation masks dilation (`dilate_segmasks.py`);
2. Data cropping (for CAC segmentation purposes, `crop_data.py`);
4. Run the CAC segmentation Deep Learning model on the cropped data (`run_inference.py`). The architecture of the Deep Learning model trained for the heart localization can be found in `cacseg_model.py`.

The CAC segmentation step can be run by executing:

```
python run_step3_cac_segmentation.py
```
```
python run_step3_cac_segmentation.py --conf ${NAME_OF_CONF_FILE}.yaml
```

By default, `config/step3_cac_segmentation.yaml` is loaded. All the tweakable parameters (e.g., path to the inferred heart segmentation data folder, output folders naming, wheter or not to export png images for quality control purposes) for this third step can be found (and modified!) in such file.

### Brief Description


## Step 4: CAC Scoring

The CAC scoring step computes the Agatston CAC score starting from the CAC segmentation resulting from our fully automated pipeline.

The CAC scoring step can be run by executing:

```
python run_step4_cac_scoring.py
```
```
python run_step4_cac_scoring.py --conf ${NAME_OF_CONF_FILE}.yaml
```

By default, `config/step4_cac_scoring.yaml ` is loaded. All the tweakable parameters (e.g., path to the data folder, where to save the CAC scoring data) for this fourth and last step can be found (and modified!) in such file.

### Brief Description


