""" Visualize training in order to detect potential overfitting and evaluate
    the training progress. """

import numpy as np
import json
import matplotlib.pyplot as plt

# global variables
network_name = 'resnet50_aug2'
TRAIN_DICT_PATH = '[PATH]/train_dict_'+network_name+'.json'
IMAGE_PATH = '[PATH]'


if __name__ == "__main__":

    with open(TRAIN_DICT_PATH) as json_file:
        train_dict = json.load(json_file)

    train_loss_list = train_dict['train_loss']
    val_loss_list = train_dict['val_loss']
    train_acc_list = train_dict['train_acc']
    val_acc_list = train_dict['val_acc']
    n_epochs = len(train_loss_list)

    x = np.linspace(1, n_epochs, n_epochs)

    plt.plot(x, np.asarray(train_loss_list), label='Train')
    plt.plot(x, np.asarray(val_loss_list), label='Validation')
    # plt.title("Curve plotted using the given points")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(IMAGE_PATH+network_name+'_loss.jpg')
    plt.savefig(IMAGE_PATH+network_name+'_loss.pdf')
    plt.show()
    plt.clf()

    plt.plot(x, np.asarray(train_acc_list)/16, label='Train')
    plt.plot(x, np.asarray(val_acc_list)/16, label='Validation')
    # plt.title("Curve plotted using the given points")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(IMAGE_PATH+network_name+'_acc.jpg')
    plt.savefig(IMAGE_PATH+network_name+'_acc.pdf')
    plt.show()
    plt.clf()
