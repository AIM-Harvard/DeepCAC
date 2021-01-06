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
import tables
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from glob import glob


def get_files(input_dir, patient_files):
  imgTrainFiles = glob(input_dir + '/*_img.nrrd')
  for imgFile in imgTrainFiles:
    mskFile = imgFile.replace('img', 'msk')
    if not os.path.exists(mskFile):
      mskFile = ''
    patient_files.append([imgFile, mskFile])

  print "Found", len(patient_files), "patients"
  return patient_files


def write_data_file(dataDir, dataFile, fileList, fill_holes, cube_size):
  print('Preprocess data')
  nrrdReader = sitk.ImageFileReader()

  outputFile = os.path.join(dataDir, dataFile + '.h5')

  if os.path.exists(outputFile):
    os.remove(outputFile)
  hdf5File = tables.open_file(outputFile, mode='w')

  pIdHdf5 = hdf5File.create_earray(hdf5File.root, 'ID', tables.StringAtom(itemsize=65), shape=(0,))
  imgHdf5 = hdf5File.create_earray(hdf5File.root, 'img', tables.FloatAtom(),
                                   shape=(0, cube_size[2], cube_size[1], cube_size[0]))
  mskHdf5 = hdf5File.create_earray(hdf5File.root, 'msk', tables.UIntAtom(),
                                   shape=(0, cube_size[2], cube_size[1], cube_size[0]))

  for fileName in fileList:
    # patientID = (os.path.basename(fileName[0])).split('_')[0]
    patientID = os.path.basename(fileName[0]).replace('_img.nrrd', '')
    print 'Process patient', patientID

    nrrdReader.SetFileName(fileName[0])
    imgSitk = nrrdReader.Execute()
    imgCube = sitk.GetArrayFromImage(imgSitk)

    if not fileName[1] == '':
      nrrdReader.SetFileName(fileName[1])
      mskSitk = nrrdReader.Execute()
      mskCube = sitk.GetArrayFromImage(mskSitk)
    else:
      mskCube = np.zeros(imgCube.shape)

    imgCube = ((np.clip(imgCube, -1024.0, 3071.0)) - 1023.5) / 2047.5
    mskCube[mskCube == 2] = 0
    mskCube[mskCube > 0] = 1

    if fill_holes:
      for sliceNr in range(len(mskCube)):
        mskCube[sliceNr] = ndimage.binary_fill_holes(mskCube[sliceNr])

    pIdHdf5.append(np.array([patientID], dtype='S65'))
    imgHdf5.append(imgCube[np.newaxis, ...])
    mskHdf5.append(mskCube[np.newaxis, ...])

  hdf5File.close()


def input_data_prep(input_dir, output_dir, run, fill_holes, final_size):
  print "\nInput data preparation:"
  print 'Loading input data from "%s"'%(input_dir)
  patient_files = []
  patient_files = get_files(input_dir, patient_files)

  if run == 'Train':
    trainFiles = patient_files[0:int(len(patient_files) * 0.7)]
    valFiles = patient_files[len(trainFiles):]
    write_data_file(output_dir, "step2_training_data", trainFiles, fill_holes, final_size)
    write_data_file(output_dir, "step2_val_data", valFiles, fill_holes, final_size)
  else:
    write_data_file(output_dir, "step2_test_data", patient_files, fill_holes, final_size)
