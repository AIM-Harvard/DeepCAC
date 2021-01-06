# Data

This folder stores all the provided files (i.e., the sample input data, and the models weights) and it is, by default, the directory where all the outputs resulting from the heart segmentation pipeline are saved.

## Directory Structure

### Input Files (Provided)

Together with all the code needed to run the DeepCAC pipeline, we provide:

* four sample subjects' CT data and the associated manual segmentation masks (heart + CAC) saved in NRRD format. These files can be found under the `raw` folder and provide a way to test if the pipeline was set-up and working correctly, as well as exemplify how the input data should be structured to be parsed correctly by the former;
* the weights of the deep learning model trained for the heart localization task (first step of the pipeline), under the `step1_heartloc/model_weights` folder;
* the weights of the deep learning model trained for the heart segmentation task (second step of the pipeline), under the `step1_heartseg/model_weights` folder;
* the weights of the deep learning model trained for the CAC segmentation task (third step of the pipeline), under the `step1_cacseg/model_weights` folder;

### Output Files

Here follows a brief description of how the DeepCAC pipeline outputs are arranged by default.

#### Heart Localization (Step 1)

The first step of the pipeline, responsible for the heart localization, produces by default the following outputs:

* the results of the preliminary preprocessing step (NRRD files) are found under `curated`;
* one quality control image (PNG file) for each of the preprocessed volumes are found under `curated_qc`;
* `model_input`
* `model_output`
* `model_output_nrrd`
* `resampled`

#### Heart Segmentation (Step 2)

The second step of the pipeline, responsible for the heart segmentation, produces by default the following outputs:

* `bbox`;
* `cropped`;
* `model_input`
* `model_output`
* `model_output_nrrd`
* `model_output_metrics`

#### CAC Segmentation (Step 3)

The third step of the pipeline, responsible for the CAC segmentation, produces by default the following outputs:

* `cropped`;
* `cropped_qc`
* `dilated`
* `model_output`

#### CAC Scoring (Step 4)

The fourth and last step of the pipeline, responsible for the CAC scoring, produces by default a CSV file `cac_score_results.csv` formatted in the following fashion:

| PatientID | Dice | CAC_calc | CAC_pred | Class_calc | Class_pred |
|-----------|------|----------|----------|------------|------------|
|  ABC123   |  -   |     -    |     -    |      -     |      -     |

where:

* `PatientID` identifies each patient using the same name scheme found in `raw`;
* if a (manual) CAC segmentation mask to compare the results of the pipeline with is provided, the Dice Coefficient for the CAC segmentation is stored under `Dice` (otherwise, the cell is set to -1);
* if a (manual) CAC segmentation mask is provided, the CAC score computed from such mask is stored under `CAC_calc` (otherwise, the cell is set to -1);
* the CAC score computed from the inferred segmentation mask is stored under`CAC_pred`;
* if a (manual) CAC segmentation mask is provided, the CAC class to which the patient belongs, computed from such mask, is stored under `Class_calc`;
*  the CAC class to which the patient belongs, computed from the inferred segmentation mask, is stored under `Class_pred`.

For instance, after running the DeepCAC pipeline using the provided sample subjects as input, such CSV file should look like:

| PatientID | Dice | CAC_calc | CAC_pred | Class_calc | Class_pred |
|-----------|------|----------|----------|------------|------------|
|    0187   |  0   |     0    |     0    |      0     |      0     |
|    0909   |0.776 | 1260.875 | 811.965  |      3     |      3     |
|    0987   |0.933 |  127.82  |  138.985 |      2     |      2     |
|    0506   | 0.77 | 318.395  |  184.8   |      3     |      2     |


## Input Data Format

Input data should be stored under the `raw` folder in the following fashion:

* for each patient, the user must create a subdirectory in `raw` (e.g., named after the patient ID - `XYZ123`);
* each patient subdirectory must contain at least the CT Series to segment the heart from (in `.nrrd` format, saved as `img.nrrd`) and, optionally, a (manual) segmentation mask (in `.nrrd` format, saved as `msk.nrrd`) to compare the segmentation resulting from the pipeline with. Whether the (manual) segmentation masks are provided or not should be specified in the scripts configuration files found under `src/conf` (specifically, setting the `has_manual_seg` to `True` or `False`).

```
data/
    |_ raw/
          |_ ABC123/
          |        |_ img.nrrd
          |        |_ msk.nrrd
          |
          |_ ABC456/
          |        |_ img.nrrd
          |        |_ msk.nrrd
          |
          ...
```

### DICOM to NRRD Conversion

In order to convert DICOM CT and RTSTRUCT Series to `.nrrd`, it is a good practice to use [Plastimatch](https://plastimatch.org), an open source software for image computation. The conversion exploiting Plastimatch can be executed directly from command line/scripted in bash: 

```
plastimatch convert --input $dicom_ct_path --output-img $ct_nrrd_path

plastimatch convert --input $dicom_rt_path --referenced-ct $dicom_ct_path 
                    --output-prefix $rt_folder --prefix-format nrrd
                    --output-ss-list $rt_struct_list_path
```


or scripted using, for example, python's `subprocess` module.

```
# convert DICOM CT to NRRD file - no resampling
bash_command = list()
bash_command += ["plastimatch", "convert"]
bash_command += ["--input", dicom_ct_path]
bash_command += ["--output-img", ct_nrrd_path]
               
# execute command
subprocess.call(bash_command)


# convert DICOM RTSTRUCT to NRRD file - no resampling
bash_command = list()
bash_command += ["plastimatch", "convert"]
bash_command += ["--input", dicom_rt_path]
bash_command += ["--referenced-ct", dicom_ct_path]
bash_command += ["--output-prefix", rt_folder]
bash_command += ["--prefix-format", 'nrrd']
bash_command += ["--output-ss-list", rt_struct_list_path]
  
# execute command
subprocess.call(bash_command)
```

For additional and more precise information, please see the [Plastimatch documentation page](https://plastimatch.org/plastimatch.html#plastimatch-convert).
