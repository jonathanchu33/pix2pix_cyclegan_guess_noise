import os
import numpy as np
from PIL import Image

IMG_DIR_PATH = '../results/test/test_latest/images' # Directory where test images are stored
normalize = True                                    # Normalize photos before calculating metric
quantizations = 50                                  # Number of even quantizations
IMG_BATCH_SIZE = 14
# palette_quantization = False # Palette quantization strategy - unused

## Metrics
rh_scores = []                    # RH
rh_grayscale_scores = []          # RH (Grayscale)
proposed_rh_scores = []           # Proposed RH
proposed_rh_grayscale_scores = [] # Proposed RH (Grayscale)
seg_scores = []                   # Segmentation Accuracy
seg_grayscale_scores = []         # Segmentation Accuracy (Grayscale)
iou_scores = []                   # IoU
iou_grayscale_scores = []         # IoU (Grayscale)

# Collect image file names from IMG_DIR_PATH
walk_dir_obj = os.walk(IMG_DIR_PATH)
root, dirs, files = next(walk_dir_obj)
image_files = sorted(files)
if '.DS_Store' in image_files:
  image_files.remove('.DS_Store')

# Calculate metric for each image. Image sets come in batches of IMG_BATCH_SIZE files
for i in range(0, len(image_files), IMG_BATCH_SIZE):
  ### (A) Prepare Images
  # Collect image names by A/B category
  imagesA, imagesB = [], []
  for j in range(0, IMG_BATCH_SIZE, 2):
    imagesA.append(image_files[i + j])
  for j in range(1, IMG_BATCH_SIZE, 2):
    imagesB.append(image_files[i + j])

  # Open images
  for i in range(len(imagesA)):
    imagesA[i] = Image.open(os.path.join(IMG_DIR_PATH, imagesA[i]))
  for i in range(len(imagesB)):
    imagesB[i] = Image.open(os.path.join(IMG_DIR_PATH, imagesB[i]))

  # Grayscale numpy arrays
  gimagesA, gimagesB = [], []
  for i in range(len(imagesA)):
    gimagesA.append(np.array(imagesA[i].convert("L")))
  for i in range(len(imagesB)):
    gimagesB.append(np.array(imagesB[i].convert("L")))

  # RGB numpy arrays
  for i in range(len(imagesA)):
    imagesA[i] = np.array(imagesA[i])
  for i in range(len(imagesB)):
    imagesB[i] = np.array(imagesB[i])

  # Normalize if specified
  if normalize:
    for i in range(len(imagesA)):
      imagesA[i] = imagesA[i]/255.0
    for i in range(len(imagesB)):
      imagesB[i] = imagesB[i]/255.0
    for i in range(len(gimagesA)):
      gimagesA[i] = gimagesA[i]/255.0
    for i in range(len(gimagesB)):
      gimagesB[i] = gimagesB[i]/255.0

  # Unpack images
  fakeA, qefakeA, qerealA, realA, recA, qerecA, snrecA = imagesA
  fakeB, qefakeB, qerealB, realB, recB, qerecB, snrecB = imagesB
  # fakeA, qefakeA, qerealA, qpfakeA, qprealA, realA, recA, qerecA, qprecA, snrecA = imagesA

  gfakeA, gqefakeA, gqerealA, grealA, grecA, gqerecA, gsnrecA = gimagesA
  gfakeB, gqefakeB, gqerealB, grealB, grecB, gqerecB, gsnrecB = gimagesB

  ### (B) Calculate metrics

  ## (B1) RH
  # RGB
  rec_loss = np.linalg.norm(recA - realA)    # Normal reconstruction loss of real image (from richer domain)
  qrec_loss = np.linalg.norm(qerecA - realA) # Reconstruction loss using quantized intermediary
  rh_scores.append(rec_loss - qrec_loss)
  # Grayscale
  grec_loss = np.linalg.norm(grecA - grealA)    # Normal reconstruction loss of real image (from richer domain)
  gqrec_loss = np.linalg.norm(gqerecA - grealA) # Reconstruction loss using quantized intermediary
  rh_grayscale_scores.append(grec_loss - gqrec_loss)

  ## (B2) Proposed RH
  # RGB
  trans_loss = np.linalg.norm(fakeA - realA) # Translation loss from one-to-many (input map from poorer domain)
  proposed_rh_scores.append(rec_loss - trans_loss)

  # Grayscale
  gtrans_loss = np.linalg.norm(gfakeA - grealA) # Translation loss from one-to-many (input map from poorer domain)
  proposed_rh_grayscale_scores.append(grec_loss - gtrans_loss)

  ## (B3) Segmentation Accuracy
  # RGB
  seg_acc = np.mean(qefakeB == qerealB)
  seg_scores.append(seg_acc)
  # Grayscale
  gseg_acc = np.mean(gqefakeB == gqerealB)
  seg_grayscale_scores.append(gseg_acc)

  ## (B4) IoU Accuracy
  # RGB
  ious = []
  segments = np.unique(qerealB)
  # if palette_quantization:
  #   # Using Palette quantization
  #   segments = np.array([[233, 233, 217],
  #                       [211, 201, 189],
  #                       [189, 181, 171],
  #                       [99, 161, 253],
  #                       [151, 191, 89],
  #                       [245, 59, -169],
  #                       [217, 163, 155],
  #                       [255, 255, 255]
  #                       ])
  for color in segments:
    intersection = np.sum(np.logical_and(qefakeB == color, qerealB == color))
    union = 256 * 256 * 3
    ious.append(intersection / union)
  iou_scores.append(np.array(ious).mean())

  # Grayscale
  gious = []
  gray_segments = np.unique(gqerealB)
  # if palette_quantization:
    # gray_segments = np.array([243, 228, 218, 203, 211, 171, 216, 255])
  for color in gray_segments:
    intersection = np.sum(np.logical_and(gqefakeB == color, gqerealB == color))
    union = 256 * 256
    gious.append(intersection / union)
  iou_grayscale_scores.append(np.array(gious).mean())

## Print scores
print('Images were', 'not' if not normalize else '\b', 'normalized.\n')

rh_scores = np.array(rh_scores)
rh_grayscale_scores = np.array(rh_grayscale_scores)
print('RH Score: ', rh_scores.mean())
print('RH (Grayscale) Score: ', rh_grayscale_scores.mean())

proposed_rh_scores = np.array(proposed_rh_scores)
proposed_rh_grayscale_scores = np.array(proposed_rh_grayscale_scores)
print('Proposed RH Score: ', proposed_rh_scores.mean())
print('Proposed RH (Grayscale) Score: ', proposed_rh_grayscale_scores.mean())

seg_scores = np.array(seg_scores)
seg_grayscale_scores = np.array(seg_grayscale_scores)
print('Segmentation Accuracy: ', seg_scores.mean())
print('Segmentation (Grayscale) Accuracy: ', seg_grayscale_scores.mean())

iou_scores = np.array(iou_scores)
iou_grayscale_scores = np.array(iou_grayscale_scores)
print('IoU Accuracy: ', iou_scores.mean())
print('IoU (Grayscale) Accuracy: ', iou_grayscale_scores.mean())
