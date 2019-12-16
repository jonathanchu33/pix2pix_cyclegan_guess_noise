import os, subprocess
import numpy as np
from PIL import Image
import argparse

IMG_BATCH_SIZE = 14

def calculate_sn(sigma):
  # Generate images with sigma noise
  print('Generate images with sigma', sigma)
  subprocess.run(['python', 'test.py',
                  '--model', MODEL_TYPE,
                  '--dataroot', 'datasets/maps',
                  '--checkpoints_dir', os.path.join('metrics', CHECKPOINTS_DIR),
                  '--results_dir', os.path.join('metrics', RESULTS_DIR),
                  '--name', MODEL_NAME,
                  '--num_test', NUM_TEST,
                  '--calculate_sn',
                  '--noise_std', str(sigma)],
                  cwd='..')

  # Collect image file names from IMG_DIR_PATH
  walk_dir_obj = os.walk(IMG_DIR_PATH)
  root, dirs, files = next(walk_dir_obj)
  image_files = sorted(files)
  if '.DS_Store' in image_files:
    image_files.remove('.DS_Store')

  # Metrics
  sn_scores = []
  sn_grayscale_scores = []

  # Calculate metrics for each image. Image sets come in batches of IMG_BATCH_SIZE files
  for i in range(0, len(image_files), IMG_BATCH_SIZE):
    ### (A) Prepare Images
    # Collect image names by A/B category
    imagesA = [image_files[i + 8], image_files[i + 12]]
    imagesB = []
    # imagesA, imagesB = [], []
    # for j in range(0, IMG_BATCH_SIZE, 2):
    #   imagesA.append(image_files[i + j])
    # for j in range(1, IMG_BATCH_SIZE, 2):
    #   imagesB.append(image_files[i + j])

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
    recA, snrecA = imagesA
    # recB, snrecB = imagesB

    grecA, gsnrecA = gimagesA
    # grecB, gsnrecB = gimagesB

    ### (B) Calculate Metric
    # RGB
    sn_loss = np.linalg.norm(snrecA - recA)
    sn_scores.append(sn_loss)

    # Grayscale
    gsn_loss = np.linalg.norm(gsnrecA - grecA)
    sn_grayscale_scores.append(gsn_loss)

  sn_scores = np.array(sn_scores)
  sn_grayscale_scores = np.array(sn_grayscale_scores)
  print('SN evaluated at', sigma, ':', sn_scores.mean())
  print('SN (Grayscale) evaluated at', sigma, ':', sn_grayscale_scores.mean())
  return sn_scores.mean(), sn_grayscale_scores.mean()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Calculate SN by integrating SN(x)dx from 0 to .2 using MC integration. ALL PATHS RELATIVE TO CURRENT DIRECTORY (metrics)')
  parser.add_argument('--normalize', action='store_true', help='Normalize images before calculating metrics.')
  parser.add_argument('--n_samples', type=int, default=10, help='Number of samples for MC integration.')
  parser.add_argument('--lower', type=float, default=0.0, help='Lower bound of integration')
  parser.add_argument('--upper', type=float, default=0.2, help='Upper bound of integration')
  parser.add_argument('--model_type', type=str, required=True, help='Type of model')
  parser.add_argument('--model_name', type=str, required=True, help='Name of saved model')
  parser.add_argument('--checkpoints_dir', type=str, required=True, help='Same as test.py --checkpoints_dir. RELATIVE TO CURRENT DIRECTORY (metrics)')
  parser.add_argument('--results_dir', type=str, required=True, help='Same as test.py --results_dir. RELATIVE TO CURRENT DIRECTORY (metrics)')
  parser.add_argument('--num_test', type=int, default=1098, help='Number of test images to consider')

  args = vars(parser.parse_args())

  normalize = args['normalize']
  N_SAMPLES = args['n_samples']
  a = args['lower']
  b = args['upper']

  MODEL_TYPE = args['model_type']
  MODEL_NAME = args['model_name']
  NUM_TEST = str(args['num_test'])
  CHECKPOINTS_DIR = args['checkpoints_dir']
  RESULTS_DIR = args['results_dir']

  IMG_DIR_PATH = os.path.join(RESULTS_DIR, MODEL_NAME, 'test_latest', 'images')


  ## Main Program
  sigmas = []
  sn_overall = []
  gsn_overall = []
  for i in range(N_SAMPLES):
    print('Sample number', i)
    sigma = np.random.uniform(a, b)
    mean, gmean = calculate_sn(sigma)
    sigmas.append(sigma)
    sn_overall.append(mean)
    gsn_overall.append(gmean)
  print('Model SN: ', np.array(sn_overall).mean())
  print('Model GSN: ', np.array(gsn_overall).mean())
  print('Sigmas sampled: ', sigmas)
