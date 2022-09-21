"""
Script to train the neural network (e.g. ResNet).
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
from pathlib import Path
import tensorflow as tf

# global variables
network_name = 'resnet50_aug2'
use_aug = True # wheter augmentations should be used on the training data
seed = 2022
continue_training = False # Continue the training process?
                        #If set to False the network will be trained from scratch
DATASET_PATH = '[PATH].zip'
MODEL_PATH = '[PATH]/checkpoint_model_'+network_name+'.pth'
OPTIMIZER_PATH = '[PATH]/checkpoint_opt_'+network_name+'.pth'
TRAIN_DICT_PATH = '[PATH]/train_dict_'+network_name+'.json'

def aug_pipeline(img_in):
    """ This augmentation pipeline will apply the stated augmentations with a
        probability to be set. Those augmentations which feature a variable for
        the power of the applied effect can be adjusted here as well. """
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
    return img

class CustomDataset(Dataset):
    """ Adjusted torch dataset to get class directly from path structure. """
    def __init__(self, path, use_aug=False):
        self.zip_path = path
        self.z = zipfile.ZipFile(self.zip_path)
        self.data_adress = [str(f) for f in sorted(self.z.namelist()) if\
                            self.is_image(f)]
        self.use_aug = use_aug

    def is_image(self, fname):
        return fname.split('.')[-1] == "jpg"

    def class_from_path(self, fname):
        # class_map = {"Negative":0, "Positive":1}
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
          img = aug_pipeline_2(img)
        img = img/255
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor, torch.tensor(label)

def train_model(model, criterion, optimizer, dataloaders, scheduler=None,\
                train_dict=None, num_epochs=20):
    if train_dict:
      train_loss_list = train_dict['train_loss']
      val_loss_list = train_dict['val_loss']
      train_acc_list = train_dict['train_acc']
      val_acc_list = train_dict['val_acc']
    else:
      train_loss_list = []
      val_loss_list = []
      train_acc_list = []
      val_acc_list = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if scheduler:
                        scheduler.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / len(dataloaders[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss,\
                                                        epoch_acc))
            if phase == 'train':
              train_loss_list.append(float(epoch_loss))
              train_acc_list.append(float(epoch_acc))
            else:
              val_loss_list.append(float(epoch_loss))
              val_acc_list.append(float(epoch_acc))
        train_dict = {'train_loss': train_loss_list,
                      'val_loss': val_loss_list,
                      'train_acc': train_acc_list,
                      'val_acc': val_acc_list}
        #save progress
        torch.save(model.state_dict(), MODEL_PATH)
        torch.save(optimizer.state_dict(), OPTIMIZER_PATH)
        with open(TRAIN_DICT_PATH, 'w') as fp:
            json.dump(train_dict, fp)
    return model, train_dict


if __name__ == "__main__":

    # look for gpu to train on
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
      raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    # load training data and make train/test split
    dataset = CustomDataset(DATASET_PATH, use_aug)
    lengths = [int(dataset.__len__()*0.8), int(dataset.__len__()*0.2)]
    print("# DEBUG: lengths: ", lengths)
    data_train, data_test = torch.utils.data.random_split(dataset, lengths,\
                                                          generator=torch\
                                                          .Generator()\
                                                          .manual_seed(seed))
    data_loaders = {'train': torch.utils.data.DataLoader(data_train,\
                                                            batch_size=16,\
                                                            shuffle=True),
                    'validation': torch.utils.data.DataLoader(data_test,\
                                                                batch_size=16,\
                                                                shuffle=True)}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # select model
    model = models.resnet50(pretrained=False).to(device)
    # model = models.resnet18(pretrained=False).to(device)

    model.fc = nn.Sequential(
                nn.Linear(2048, 128), # this is used for ResNet50
                # nn.Linear(512, 128), # this is used for ResNet18
                nn.ReLU(inplace=True),
                nn.Linear(128, 2)).to(device) # softmax in loss function


    model = model.float()

    # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters()) # for ResNet
    scheduler = None
    # optimizer = optim.SGD(model.parameters(), lr=1, momentum=0.9) # for VGG
    # scheduler_param = [5, 0.5] # [step_size, gamma]
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    if continue_training:
        model.load_state_dict(torch.load(MODEL_PATH))
        optimizer.load_state_dict(torch.load(OPTIMIZER_PATH))
    with open(TRAIN_DICT_PATH) as json_file:
        train_dict = json.load(json_file)
    else:
        train_dict = None

    # execute training
    model_trained, train_dict = train_model(model, criterion, optimizer,\
                                            data_loaders, scheduler=scheduler,\
                                            train_dict=train_dict, num_epochs=20)

    # save model
    torch.save(model_trained.state_dict(), MODEL_PATH)
    # save training progress
    with open(TRAIN_DICT_PATH, 'w') as fp:
        json.dump(train_dict, fp)
