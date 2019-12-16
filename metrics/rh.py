import os
import numpy as np
from PIL import Image

IMG_DIR_PATH = '../results/test/test_latest/images'

normalize = False

walk_dir_obj = os.walk(IMG_DIR_PATH)
root, dirs, files = next(walk_dir_obj)
image_files = sorted(files)
if '.DS_Store' in image_files:
  image_files.remove('.DS_Store')

rh_scores = []
rh_grayscale_scores = []

# Image sets come in batches of 10 files
for i in range(0, len(image_files), 14):
  fakeA, _, _, realA, recA, qrecA, _ = image_files[i], 1, 1, image_files[i + 6], image_files[i + 8], image_files[i + 10], 1
  fakeA, _, _, realA, recA, qrecA, _ = Image.open(os.path.join(IMG_DIR_PATH, fakeA)), 1, 1, Image.open(os.path.join(IMG_DIR_PATH, realA)), Image.open(os.path.join(IMG_DIR_PATH, recA)), Image.open(os.path.join(IMG_DIR_PATH, qrecA)), 1
  # Grayscale
  gfakeA, grealA, grecA, gqrecA = fakeA.convert("L"), realA.convert("L"), recA.convert("L"), qrecA.convert("L")
  gfakeA, grealA, grecA, gqrecA = np.array(gfakeA), np.array(grealA), np.array(grecA), np.array(gqrecA)
  # RGB
  fakeA, realA, recA, qrecA = np.array(fakeA), np.array(realA), np.array(recA), np.array(qrecA)
  # print(fakeA, realA, recA)
  # print(np.unique(fakeA, axis=1))
  # print(np.unique(realA, axis=1).shape)
  # print(np.unique(recA, axis=1).shape)

  if normalize:
    gfakeA, grealA, grecA, gqrecA = gfakeA/255.0, grealA/255.0, grecA/255.0, gqrecA/255.0
    fakeA, realA, recA, qrecA = fakeA/255.0, realA/255.0, recA/255.0, qrecA/255.0

  # RGB
  rec_loss = np.linalg.norm(recA - realA)        # Normal reconstruction loss
  qrec_loss = np.linalg.norm(qrecA - realA)  # Reconstruction loss using quantized intermediary
  rh_scores.append(rec_loss - qrec_loss)

  # Grayscale
  grec_loss = np.linalg.norm(grecA - grealA)        # Normal reconstruction loss
  gqrec_loss = np.linalg.norm(gqrecA - grealA)  # Reconstruction loss using quantized intermediary
  rh_grayscale_scores.append(grec_loss - gqrec_loss)

rh_scores = np.array(rh_scores)
rh_grayscale_scores = np.array(rh_grayscale_scores)
print('RH Score: ', rh_scores.mean())
print('RH (Grayscale) Score: ', rh_grayscale_scores.mean())
