""" Use the trained network in a sliding window approach to detect cracks in
    larger images. """

import numpy as np
import zipfile
from PIL import Image
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
import cv2
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl

# global variables
network_name = 'resnet18'
model = 'resnet18'
DATASET_PATH = '[PATH]/CRACK_forMIN.zip'
MODEL_PATH = '[PATH]/checkpoint_model_'+network_name+'.pth'
IMAGE_PATH = '[PATH]'
# Images_PATH = '[PATH]'
seed = 2022
torch.manual_seed(seed)


class CustomDataset(Dataset):
    def __init__(self, path, use_aug=False):
        self.zip_path = path
        self.z = zipfile.ZipFile(self.zip_path)
        self.data_adress = [str(f) for f in sorted(self.z.namelist()) if\
                            self.is_image(f)]
        self.use_aug = use_aug

    def is_image(self, fname):
        return fname.split('.')[-1] == "jpg"

    def __len__(self):
        return len(self.data_adress)

    def __getitem__(self, idx):
        img_adress = self.data_adress[idx]
        label = 1#self.class_from_path(img_adress)
        data = self.z.read(img_adress)
        dataEnc = io.BytesIO(data)
        img = np.asarray(Image.open(dataEnc))
        if self.use_aug:
          img = aug_pipeline(img)
        img = img/255
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor, torch.tensor(label)

def slice_image(image):
    """ +53 pixel each time (15 times), while original image is tronsformed
        to size 1022 """
    image_list = []
    for i in range(16):
        for j in range(16):
            image_list.append(image[:,:,0+53*i:277+53*i,0+53*j:277+53*j])
    return image_list

def make_overlay(output_list):
    overlay_raw = np.zeros([1022,1022])
    overlay_denominator = np.zeros([1022,1022])
    iter = 0
    for i in range(16):
        for j in range(16):
            output = outputs_list[iter]
            value = output[0,0]
            overlay_raw[0+53*i:227+53*i,0+53*j:227+53*j] += value
            overlay_denominator[0+53*i:227+53*i,0+53*j:227+53*j] += 1
            iter += 1
    overlay = overlay_raw/overlay_denominator
    return overlay

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap

def applyCustomColorMap(im_gray) :
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    #Red
    # lut[:, 0, 0] = [255]*256
    lut[:, 0, 0] = list(range(256))
    #Green
    # lut[:, 0, 1] = [255]*256
    lut[:, 0, 1] = [155]*256
    #Blue
    # lut[:, 0, 2] = list(range(256))
    lut[:, 0, 2] = [155]*256
    #Apply custom colormap through LUT
    im_color = cv2.LUT(im_gray, lut)
    return im_color


if __name__ == "__main__":

    # load data
    dataset = CustomDataset(DATASET_PATH)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(" DEBUG: device: ", device)

    if model == 'resnet50':
      model = models.resnet50(pretrained=False).to(device)
      model.fc = nn.Sequential(
                  nn.Linear(2048, 128),
                  nn.ReLU(inplace=True),
                  nn.Linear(128, 2),
                  nn.Softmax()).to(device) #added here
    elif model == 'resnet18':
      model = models.resnet18(pretrained=False).to(device)
      model.fc = nn.Sequential(
                  nn.Linear(512, 128),
                  nn.ReLU(inplace=True),
                  nn.Linear(128, 2),
                  nn.Softmax()).to(device) #added here
    else:
      print('No valid model type given.')

    model = model.float()
    if torch.cuda.is_available():
      model.load_state_dict(torch.load(MODEL_PATH))
    else:
      model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    # feed big images slice whise through the network
    phase = 'validation'

    for i in range(30):
        inputs, labels = next(iter(data_loader))
        image_list = slice_image(inputs)

        outputs_list = []
        for inputs_part in image_list:
            inputs_part = inputs_part.to(device)
            model.eval()
            outputs = model(inputs_part)
            outputs_list.append(outputs.cpu().detach().numpy())

        # combine big input image with slice whise network outputs
        overlay = make_overlay(outputs_list)
        image = inputs[0,:,0:1022,0:1022].cpu().detach().numpy().transpose(1,2,0)
        img = cv2.resize(image,(1022,1022))
        heatmap = cv2.applyColorMap(np.uint8(255*overlay), cv2.COLORMAP_JET)
        img = heatmap * 0.3 + img

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

        im = grid[0].imshow(img)
        grid[0].axis('off')
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cbar = grid[1].figure.colorbar(
                        mpl.cm.ScalarMappable(norm=norm, cmap='jet'),
                        ax=grid[1], alpha=0.3)
        grid[1].axis('off')
        # plt.show()
        save_name = IMAGE_PATH + 'sw_' + str(i) + '.png'
        plt.savefig(save_name)
