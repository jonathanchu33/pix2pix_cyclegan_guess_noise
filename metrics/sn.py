import os, subprocess
import numpy as np
from PIL import Image

## Run this script from the metrics directory.
IMG_DIR_PATH = '../results/test/test_latest/images'
MODEL ='cycle_gan_noisy'

normalize = True

def calculate_sn(sigma):
  print('Generate images with sigma', sigma)
  subprocess.run(['python', 'test.py', '--model', MODEL, '--dataroot', 'datasets/maps', '--name', 'test', '--num_test', '75', '--calculate_sn', '--noise_std', str(sigma)], cwd='..')
  # 'python ../test.py --model cycle_gan_noisy --dataroot datasets/maps --name test --num_test 75 --gpu_ids -1 --calculate_sn'
  walk_dir_obj = os.walk(IMG_DIR_PATH)
  root, dirs, files = next(walk_dir_obj)
  image_files = sorted(files)
  if '.DS_Store' in image_files:
    image_files.remove('.DS_Store')

  sn_scores = []
  sn_grayscale_scores = []

  # Image sets come in batches of 8 files
  for i in range(1, len(image_files), 8):
    fakeA, realA, recA, snrecA = image_files[i], image_files[i + 2], image_files[i + 4], image_files[i + 6]
    fakeA, realA, recA, snrecA = Image.open(os.path.join(IMG_DIR_PATH, fakeA)), Image.open(os.path.join(IMG_DIR_PATH, realA)), Image.open(os.path.join(IMG_DIR_PATH, recA)), Image.open(os.path.join(IMG_DIR_PATH, snrecA))
    # Grayscale
    gfakeA, grealA, grecA, gsnrecA = fakeA.convert("L"), realA.convert("L"), recA.convert("L"), snrecA.convert("L")
    gfakeA, grealA, grecA, gsnrecA = np.array(gfakeA), np.array(grealA), np.array(grecA), np.array(gsnrecA)
    # RGB
    fakeA, realA, recA, snrecA = np.array(fakeA), np.array(realA), np.array(recA), np.array(snrecA)

    if normalize:
      gfakeA, grealA, grecA, gsnrecA = gfakeA/255.0, grealA/255.0, grecA/255.0, gsnrecA/255.0
      fakeA, realA, recA, snrecA = fakeA/255.0, realA/255.0, recA/255.0, snrecA/255.0

    # RGB
    sn_loss = np.linalg.norm(snrecA - recA)
    sn_scores.append(sn_loss)

    # Grayscale
    gsn_loss = np.linalg.norm(gsnrecA - grecA)
    sn_grayscale_scores.append(gsn_loss)

  sn_scores = np.array(sn_scores)
  sn_grayscale_scores = np.array(sn_grayscale_scores)
  print('Proposed SN Score: ', sn_scores.mean())
  print('Proposed SN (Grayscale) Score: ', sn_grayscale_scores.mean())
  return sn_scores.mean(), sn_grayscale_scores.mean()


N_SAMPLES = 10
sigmas = []
sn_overall = []
gsn_overall = []
for i in range(N_SAMPLES):
  sigma = np.random.uniform(0.0, 2.0)
  mean, gmean = calculate_sn(sigma)
  sigmas.append(sigma)
  sn_overall.append(mean)
  gsn_overall.append(gmean)
print('Model SN: ', np.array(sn_overall).mean())
print('Model GSN: ', np.array(gsn_overall).mean())
print('Sigmas sampled: ', sigmas)
