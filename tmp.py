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
# def init_class_order():
#     """The function to initialize the class order.
#     Returns:
#       order: an array for the class order
#       order_list: a list for the class order
#     """
#     # Set the random seed according to the config
#     np.random.seed(1993)
#     # Set the name for the class order file
#
#     order = np.arange(10)
#     np.random.shuffle(order)
#     # Transfer the array to a list
#     order_list = list(order)
#     # Print the class order
#     print(order_list)
#     return order, order_list
# init_class_order()
elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
# Load previous FC weights, transfer them from GPU to CPU
old_embedding_norm = b1_model.fc.fc1.weight.data.norm(dim=1, keepdim=True)
average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).to('cpu').type(torch.DoubleTensor)
# tg_feature_model is b1_model without the FC layer
tg_feature_model = nn.Sequential(*list(b1_model.children())[:-1])
# Get the shape of the feature inputted to the FC layers, i.e., the shape for the final feature maps
num_features = b1_model.fc.in_features
# Intialize the new FC weights with zeros
novel_embedding = torch.zeros((self.args.nb_cl, num_features))
for cls_idx in range(iteration * self.args.nb_cl, (iteration + 1) * self.args.nb_cl):
    # Get the indexes of samples for one class
    cls_indices = np.array([i == cls_idx for i in map_Y_train])
    # Check the number of samples in this class
    assert (len(np.where(cls_indices == 1)[0]) <= dictionary_size)
    # Set a temporary dataloader for the current class
    current_eval_set = merge_images_labels(X_train[cls_indices], np.zeros(len(X_train[cls_indices])))
    self.evalset.imgs = self.evalset.samples = current_eval_set
    evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                                             shuffle=False, num_workers=2)
    num_samples = len(X_train[cls_indices])
    # Compute the feature maps using the current model
    cls_features = compute_features(self.args, self.fusion_vars, b1_model, b2_model, \
                                    tg_feature_model, is_start_iteration, evalloader, num_samples, num_features)
    # Compute the normalized feature maps
    norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
    # Update the FC weights using the imprint weights, i.e., the normalized averged feature maps
    cls_embedding = torch.mean(norm_features, dim=0)
    novel_embedding[cls_idx - iteration * self.args.nb_cl] = F.normalize(cls_embedding, p=2,
                                                                         dim=0) * average_old_embedding_norm
# Transfer all weights of the model to GPU
b1_model.to(self.device)
b1_model.fc.fc2.weight.data = novel_embedding.to(self.device)