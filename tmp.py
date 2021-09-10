import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import time
import os
import os.path as osp
import sys
import copy
import argparse
from PIL import Image
try:
    import cPickle as pickle
except:
    import pickle
import math
import utils.misc
import models.modified_resnet_cifar as modified_resnet_cifar
import models.modified_resnetmtl_cifar as modified_resnetmtl_cifar
import models.modified_resnet as modified_resnet
import models.modified_resnetmtl as modified_resnetmtl
import models.modified_linear as modified_linear
from utils.imagenet.utils_dataset import split_images_labels
from utils.imagenet.utils_dataset import merge_images_labels
from utils.incremental.compute_features import compute_features
from utils.incremental.compute_accuracy import compute_accuracy
from utils.misc import process_mnemonics
from torchvision.datasets import MNIST
import warnings
def init_class_order():
    """The function to initialize the class order.
    Returns:
      order: an array for the class order
      order_list: a list for the class order
    """
    # Set the random seed according to the config
    np.random.seed(1993)
    # Set the name for the class order file

    order = np.arange(10)
    np.random.shuffle(order)
    # Transfer the array to a list
    order_list = list(order)
    # Print the class order
    print(order_list)
    return order, order_list

init_class_order()