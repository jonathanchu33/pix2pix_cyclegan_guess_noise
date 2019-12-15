import os
import numpy as np
from PIL import Image

IMG_DIR_PATH = '.'

normalize = True

walk_dir_obj = os.walk(IMG_DIR_PATH)
root, dirs, files = next(walk_dir_obj)
image_files = sorted(files)
if '.DS_Store' in image_files:
  image_files.remove('.DS_Store')

rh_scores = []
rh_grayscale_scores = []

# Image sets come in batches of 6 files
for i in range(0, len(image_files), 6):
  fakeA, realA, recA = image_files[i], image_files[i + 2], image_files[i + 4]
  fakeA, realA, recA = Image.open(fakeA), Image.open(realA), Image.open(recA)
  # Grayscale
  gfakeA, grealA, grecA = fakeA.convert("L"), realA.convert("L"), recA.convert("L")
  gfakeA, grealA, grecA = np.array(gfakeA), np.array(grealA), np.array(grecA)
  # RGB
  fakeA, realA, recA = np.array(fakeA), np.array(realA), np.array(recA)

  if normalize:
    gfakeA, grealA, grecA = gfakeA/255.0, grealA/255.0, grecA/255.0
    fakeA, realA, recA = fakeA/255.0, realA/255.0, recA/255.0

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
