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
import numpy as np
from glob import glob


def load_test_data(inputFolder, mask=False):
  #print 'Loading test data'

  patientList = []
  testDataClc = []

  patientFiles = glob(inputFolder+'/*img.npy')
  for patientFile in patientFiles:
    patientID = os.path.basename(patientFile).replace('_img.npy', '')
    patientList.append(patientID)
  #print 'Found', len(patientFiles), 'files for clc and hlt patients'

  for patientID in patientList:

    imgFile = os.path.join(inputFolder, patientID + '_img.npy')
    if not os.path.exists(imgFile):
      print 'WARNING - img not found for patient', patientID
      continue
    try:
      img = np.load(imgFile)
    except Exception as e:
      print(e)
      continue

    if mask:
      mskFile = os.path.join(inputFolder, patientID + '_msk.npy')

      if not os.path.exists(mskFile):
        print 'WARNING - msk not found for patient', patientID
        continue
      try:
        msk = np.load(mskFile)
      except Exception as e:
        print(e)
        continue
      msk[msk > 0.9] = 1
      msk[msk < 1] = 0
    else:
      msk = np.zeros(img.shape)

    testDataClc.append([patientID, img, msk])

  #print 'Loaded', len(testDataClc), 'patients for testing'
  return testDataClc
