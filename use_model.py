""" Use the trained model to determine wheter a image shows a crack or not.
    Furthermore, a saliency map can be obtained to get insights on the
    decision making process of the network. """

import numpy as np
import zipfile
from PIL import Image
import io
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
from torch.nn import ReLU
from mpl_toolkits.axes_grid1 import ImageGrid

# global variables
shadow = True
network_name = 'resnet18_aug'
model = 'resnet18'
DATASET_PATH = '[PATH]/[dataset].zip'
RESULT_PATH = '[PATH]'
Images_PATH = '[PATH]'
seed = 2022
torch.manual_seed(seed)

class CustomDataset(Dataset):
    def __init__(self, path):
        self.zip_path = path
        self.z = zipfile.ZipFile(self.zip_path)
        self.data_adress = [str(f) for f in sorted(self.z.namelist()) if\
                            self.is_image(f)]

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
        img = np.asarray(Image.open(dataEnc))/255
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor, torch.tensor(label)


class GuidedBackprop():
    """ Produces gradients generated with guided back propagation from the given
        image. """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.children())[0]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """ Updates relu activation functions so that it only returns positive
            gradients. """
        def relu_hook_function(module, grad_in, grad_out):
            """ If there is a negative gradient, changes it to zero """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function
        for module in self.model.modules():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == "__main__":
    # load data
    dataset = CustomDataset(DATASET_PATH)
    lengths = [int(dataset.__len__()*0.8), int(dataset.__len__()*0.2)]
    data_train, data_test = torch.utils.data.random_split(dataset, lengths,\
                                                          generator=torch.Generator()\
                                                          .manual_seed(seed))
    data_loaders = {'train': torch.utils.data.DataLoader(data_train, batch_size=16,\
                                                            shuffle=True),
                    'validation': torch.utils.data.DataLoader(data_test,\
                                                                batch_size=16,\
                                                                shuffle=True)}

    # setup model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    phase = 'validation'
    dataloader = data_loaders[phase]

    inputs, labels = next(iter(dataloader))

    # plot images raw
    for i in range(0,16):
        save_name = Images_PATH + str(i) + '.png'
        single_input = inputs[i].reshape(1,3,227,227).numpy()[0].transpose(1,2,0)
        plt.imshow(single_input)
        plt.axis('off')
        plt.savefig(save_name)
        plt.clf()

    # use images not used for training
    phase = 'validation'
    dataloader = data_loaders[phase]

    # get network results with saliency map
    inputs, labels = next(iter(dataloader))

    for i in range(0,16):
        single_input = inputs[i].reshape(1,3,227,227)
        single_label = labels[i]

        device = 'cpu'

        single_input = single_input.to(device)
        single_label = single_label.to(device)

        single_input.requires_grad_(True)

        model = model.to(device)

        guided_bp = GuidedBackprop(model)
        result = guided_bp.generate_gradients(single_input, single_label)
        result = np.mean(result, axis=0)
        result = result/np.amax(np.abs(result))

        result_normalized = -1*(NormalizeData(result)-1)

        saliency_display = saliency[i].cpu().numpy()
        saliency_display = saliency_display/np.amax(np.abs(saliency_display))
        saliency_normalized = NormalizeData(saliency_display)

        # Visualize the image and the guided backpropagation map
        fig = plt.figure()

        grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                      nrows_ncols=(1,3),
                      axes_pad=0.15,
                      share_all=True,
                      cbar_location="right",
                      cbar_mode="single",
                      cbar_size="7%",
                      cbar_pad=0.15,
                      )

        im1 = grid[0].imshow(single_input[0].cpu().detach().numpy()\
                .transpose(1, 2, 0))
        grid[0].axis('off')
        im2 = grid[1].imshow(result_normalized.transpose(0,1), cmap='hot',\
                                vmin=0, vmax=1)
        grid[1].axis('off')
        im3 = grid[2].imshow(saliency_normalized, cmap='hot')
        grid[2].axis('off')
        # fig.suptitle('Image and associated guided backpropagation map')

        # Colorbar
        ax = grid[1]
        ax.cax.colorbar(im2)
        ax.cax.toggle_label(True)

        image_name = 'saliency_map_' + network_name + '_' + '_' + str(i)
        # plt.show()
        plt.savefig(RESULT_PATH+image_name+'.pdf')
        plt.savefig(RESULT_PATH+image_name+'.png')
        plt.clf()
