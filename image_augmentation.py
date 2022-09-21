"""
Comment
"""

import numpy as np
import zipfile
import json
from PIL import Image
import PIL
import io
import torch
from torch.utils.data import Dataset, DataLoader
# %matplotlib inline
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
import random
import copy

# mount data set
from google.colab import output
from google.colab import drive

drive.mount('/content/drive')

shadow = False

# network_name = 'resnet18'
# # network_name = 'resnet18_aug'
# # network_name = 'resnet18_aug2'
# model = 'resnet18'
# # newtwork_name = 'resnet50'
# # network_name = 'resnet50_aug'
# # network_name = 'resnet50_aug2'
# # model = 'resnet50'

if shadow:
  DATASET_PATH = '/content/drive/MyDrive/RP2/Images for classification_shadows.zip'
  add_name = 'shadow'
else:
  DATASET_PATH = '/content/drive/MyDrive/RP2/Images for classification_original.zip'
  add_name = 'no_shadow'
# MODEL_PATH = '/content/drive/MyDrive/RP2/remote_folder/Crack_detection/result/checkpoint_model_'+network_name+'.pth'

RESULT_PATH = '/content/drive/MyDrive/RP2/results_final/augmentations/'

def aug_pipeline(img_in):
  img = copy.deepcopy(img_in)
  probability = {'90degree':3/4,
                 'noise':0.5,
                 'horizontal_flip':0.5,
                 'brightness':1,
                 'contrast':1,
                 'sharpness':0.5}
  random_threshold = random.uniform(0,1)
  if probability['90degree']>random_threshold:
    random_int = random.randint(0,2)
    if random_int == 0:
      img = np.rot90(img, k=1)
    elif random_int == 1:
      img = np.rot90(img, k=2)
    else:
      img = np.rot90(img, k=3)
  random_threshold = random.uniform(0,1)
  if probability['noise']>random_threshold:
    #apply noise (salt and pepper noise)
    prob = 0.25
    black = np.array([0, 0, 0], dtype='uint8')
    white = np.array([255, 255, 255], dtype='uint8')
    probs = np.random.random(img.shape[:2])
    img[probs < (prob / 2)] = black
    img[probs > 1 - (prob / 2)] = white
  random_threshold = random.uniform(0,1)
  if probability['horizontal_flip']>random.uniform(0,1):
    img = np.flip(img, axis=1)
  img = Image.fromarray(img.astype('uint8')).convert('RGBA')
  random_threshold = random.uniform(0,1)
  if probability['brightness']>random_threshold:
    factor = 0.5 + random.uniform(0,1)
    enhancer = PIL.ImageEnhance.Brightness(img)
    img = enhancer.enhance(factor)
  random_threshold = random.uniform(0,1)
  if probability['contrast']>random_threshold:
    factor = 0.5 + random.uniform(0,1)
    enhancer = PIL.ImageEnhance.Contrast(img)
    img = enhancer.enhance(factor)
  random_threshold = random.uniform(0,1)
  if probability['sharpness']>random_threshold:
    factor = 0.05 + random.uniform(0,0.5)
    enhancer = PIL.ImageEnhance.Sharpness(img)
    img = enhancer.enhance(factor)
  img = img.convert('RGB')
  img = np.array(img)
  # print("# DEBUG: img.shape: ", img.shape)
  return img

class CustomDataset(Dataset):
    def __init__(self, path, use_aug=True):
        self.zip_path = path
        self.z = zipfile.ZipFile(self.zip_path)
        self.data_adress = [str(f) for f in sorted(self.z.namelist()) if\
                            self.is_image(f)]
        print("# DEBUG: len(self.data_adress): ", len(self.data_adress))
        self.use_aug = use_aug

    def is_image(self, fname):
        return fname.split('.')[-1] == "jpg"

    def class_from_path(self, fname):
        """ class_map = {"Negative":0, "Positive":1} """
        if fname.split('/')[-2] == "Negative":
            label = 0
        else:
            label = 1
        return label

    def __len__(self):
        return len(self.data_adress)

    def __getitem__(self, idx):
        img_adress = self.data_adress[idx]
        label = self.class_from_path(img_adress)
        data = self.z.read(img_adress)
        dataEnc = io.BytesIO(data)
        img = np.asarray(Image.open(dataEnc))
        if self.use_aug:
          img_aug = aug_pipeline(img)
        # img = img/255
        # img_tensor = torch.from_numpy(img).type(torch.DoubleTensor)
        # img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
        # print("# DEBUG: img_tensor.dtype: ", img_tensor.dtype)
        # img_tensor = img_tensor.permute(2, 0, 1)
        # return img_tensor, torch.tensor(label)
        return img, img_aug

from mpl_toolkits.axes_grid1 import ImageGrid

dataset = CustomDataset(DATASET_PATH)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

for i in range(10):
  img_original, img_aug = next(iter(data_loader))
  img_original = img_original[0]
  img_aug = img_aug[0]

  fig = plt.figure()
  grid = ImageGrid(fig, 111,#212,
                  nrows_ncols = (1, 2),
                  axes_pad = 0.1,
                  label_mode = "L",
                  share_all = False#,
                  # cbar_location="right",
                  # cbar_mode="single",
                  # cbar_size="7%",
                  # cbar_pad="7%",
                  # aspect = True
                  )

  im = grid[0].imshow(img_original)
  grid[0].axis('off')
  im = grid[1].imshow(img_aug)
  grid[1].axis('off')

  # plt.show()
  save_name = RESULT_PATH + 'aug_' + str(i) + '.png'
  plt.savefig(save_name)
  plt.clf()
