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
import heartseg_model
import matplotlib.pyplot as plt


def save_png(patientID, output_dir_png, img, msk, pred):
  maskIndicesMsk = np.where(msk != 0)
  if len(maskIndicesMsk) == 0:
    trueBB = [np.min(maskIndicesMsk[0]), np.max(maskIndicesMsk[0]),
              np.min(maskIndicesMsk[1]), np.max(maskIndicesMsk[1]),
              np.min(maskIndicesMsk[2]), np.max(maskIndicesMsk[2])]
    cen = [trueBB[0] + (trueBB[1] - trueBB[0]) / 2,
           trueBB[2] + (trueBB[3] - trueBB[2]) / 2,
           trueBB[4] + (trueBB[5] - trueBB[4]) / 2]
  else:
    cen = [int(len(img) / 2), int(len(img) / 2), int(len(img) / 2)]

  pred[pred > 0.5] = 1
  pred[pred < 1] = 0

  fig, ax = plt.subplots(2, 3, figsize=(32, 16))
  ax[0, 0].imshow(img[cen[0], :, :], cmap='gray')
  ax[0, 1].imshow(img[:, cen[1], :], cmap='gray')
  ax[0, 2].imshow(img[:, :, cen[2]], cmap='gray')

  ax[0, 0].imshow(msk[cen[0], :, :], cmap='jet', alpha=0.4)
  ax[0, 1].imshow(msk[:, cen[1], :], cmap='jet', alpha=0.4)
  ax[0, 2].imshow(msk[:, :, cen[2]], cmap='jet', alpha=0.4)

  ax[1, 0].imshow(img[cen[0], :, :], cmap='gray')
  ax[1, 1].imshow(img[:, cen[1], :], cmap='gray')
  ax[1, 2].imshow(img[:, :, cen[2]], cmap='gray')

  ax[1, 0].imshow(pred[cen[0], :, :], cmap='jet', alpha=0.4)
  ax[1, 1].imshow(pred[:, cen[1], :], cmap='jet', alpha=0.4)
  ax[1, 2].imshow(pred[:, :, cen[2]], cmap='jet', alpha=0.4)

  fileName = os.path.join(output_dir_png, patientID + '_' + ".png")
  plt.savefig(fileName)
  plt.close(fig)

## ----------------------------------------
## ----------------------------------------

def run_inference(model_weights_dir_path, data_dir, output_dir,
                  weights_file_name, export_png, final_size, training_size, down_steps):

  print "\nDeep Learning model inference using 4xGPUs:" 
  
  mgpu = 4

  output_dir_npy = os.path.join(output_dir, 'npy')
  output_dir_png = os.path.join(output_dir, 'png')
  if not os.path.exists(output_dir_npy):
    os.mkdir(output_dir_npy)
  if export_png and not os.path.exists(output_dir_png):
    os.mkdir(output_dir_png)

  inputShape = (training_size[2], training_size[1], training_size[0], 1)
  model = heartseg_model.getUnet3d(down_steps=down_steps, input_shape=inputShape, mgpu=mgpu, ext=True)

  print 'Loading saved model from "%s"'%(model_weights_dir_path)
  weights_file = os.path.join(model_weights_dir_path, weights_file_name)
  model.load_weights(weights_file)

  test_file = "step2_test_data.h5"
  testFileHdf5 = tables.open_file(os.path.join(data_dir, test_file), "r")

  testDataRaw = []
  for i in range(len(testFileHdf5.root.ID)):
    patientID = testFileHdf5.root.ID[i]
    img = testFileHdf5.root.img[i]
    msk = testFileHdf5.root.msk[i]
    testDataRaw.append([patientID, img, msk])

  numData = len(testDataRaw)
  imgsTrue = np.zeros((numData, training_size[2], training_size[1], training_size[0]), dtype=np.float64)
  msksTrue = np.zeros((numData, training_size[2], training_size[1], training_size[0]), dtype=np.float64)

  for i in xrange(0, len(testDataRaw), mgpu):
    imgTest = np.zeros((mgpu, training_size[2], training_size[1], training_size[0]), dtype=np.float64)

    for j in range(mgpu):
      patientIndex = min(len(testDataRaw) - 1, i + j)

      patientID = testDataRaw[patientIndex][0]
      print 'Processing patient', patientID
      # Store data for score calculation
      imgsTrue[patientIndex, 0:img.shape[0], :, :] = testDataRaw[patientIndex][1]
      msksTrue[patientIndex, 0:img.shape[0], :, :] = testDataRaw[patientIndex][2]
      imgTest[j, 0:img.shape[0], :, :] = testDataRaw[patientIndex][1]
    msksPred = model.predict(imgTest[:, :, :, :, np.newaxis])

    for j in range(mgpu):
      patientIndex = min(len(testDataRaw) - 1, i + j)

      patientID = testDataRaw[patientIndex][0]
      np.save(os.path.join(output_dir_npy, patientID + '_pred'),
              [[patientID],
               imgsTrue[patientIndex, 0:final_size[2], :, :],
               msksTrue[patientIndex, 0:final_size[2], :, :],
               msksPred[j, 0:final_size[2], :, :, 0]])

    if export_png:
      for j in range(mgpu):
        patientIndex = min(len(testDataRaw) - 1, i + j)
        patientID = testDataRaw[patientIndex][0]
        save_png(patientID, output_dir_png, imgsTrue[patientIndex, 0:final_size[2], :, :],
                 msksTrue[patientIndex, 0:final_size[2], :, :], msksPred[j, 0:final_size[2], :, :, 0])
