import os
import numpy as np
from PIL import Image

IMG_DIR_PATH = '../results/test/test_latest/images'

normalize = True

walk_dir_obj = os.walk(IMG_DIR_PATH)
root, dirs, files = next(walk_dir_obj)
image_files = sorted(files)
if '.DS_Store' in image_files:
  image_files.remove('.DS_Store')

rh_scores = []
rh_grayscale_scores = []

# Image sets come in batches of 14 files
for i in range(0, len(image_files), 14):
  fakeA, _, _, realA, recA, qrecA, _ = image_files[i], 1, 1, image_files[i + 6], image_files[i + 8], image_files[i + 10], 1
  fakeA, _, _, realA, recA, qrecA, _ = Image.open(os.path.join(IMG_DIR_PATH, fakeA)), 1, 1, Image.open(os.path.join(IMG_DIR_PATH, realA)), Image.open(os.path.join(IMG_DIR_PATH, recA)), Image.open(os.path.join(IMG_DIR_PATH, qrecA)), 1
  # Grayscale
  gfakeA, grealA, grecA, gqrecA = fakeA.convert("L"), realA.convert("L"), recA.convert("L"), qrecA.convert("L")
  gfakeA, grealA, grecA, gqrecA = np.array(gfakeA), np.array(grealA), np.array(grecA), np.array(gqrecA)
  # RGB
  fakeA, realA, recA, qrecA = np.array(fakeA), np.array(realA), np.array(recA), np.array(qrecA)

  if normalize:
    gfakeA, grealA, grecA, gqrecA = gfakeA/255.0, grealA/255.0, grecA/255.0, gqrecA/255.0
    fakeA, realA, recA, qrecA = fakeA/255.0, realA/255.0, recA/255.0, qrecA/255.0

  # RGB
  rec_loss = np.linalg.norm(recA - realA)    # Rec loss of real image (from richer domain)
  trans_loss = np.linalg.norm(fakeA - realA) # Trans loss from one-to-many (input map from poorer domain)
  rh_scores.append(rec_loss - trans_loss)

  # Grayscale
  grec_loss = np.linalg.norm(grecA - grealA)    # Rec loss of real image (from richer domain)
  gtrans_loss = np.linalg.norm(gfakeA - grealA) # Trans loss from one-to-many (input map from poorer domain)
  rh_grayscale_scores.append(grec_loss - gtrans_loss)

rh_scores = np.array(rh_scores)
rh_grayscale_scores = np.array(rh_grayscale_scores)
print('Proposed RH Score: ', rh_scores.mean())
print('Proposed RH (Grayscale) Score: ', rh_grayscale_scores.mean())
