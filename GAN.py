#!/usr/bin/env python3
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import datasets, transforms


# Set random seed for reproducibility
manualSeed = 4717300
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

# Get gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# HyperParameters
nz = 100 # Latent feature size
ngf = 64 # Size of feature maps in generator
ndf = 64 # Size of feature maps in discriminator
num_epochs = 5 # Number of training epochs
lr = 0.0002 # Learning rate for optimizers
beta1 = 0.5 # Beta1 hyperparameter for Adam optimizers


# Load data into RAM
dataroot = "data/keras_png_slices_data"

# Number of workers for dataloader
# workers = 2
channels = 1 # Only greyscale

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 256
img_shape = (channels, image_size, image_size)
ngpu = 1

# Create a class to hold the data
class GANDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load all images into a data loader
test_data = GANDataset(img_dir='data/keras_png_slices_data/keras_png_slices_test', transform=transform)
train_data = GANDataset(img_dir='data/keras_png_slices_data/keras_png_slices_train', transform=transform)	
validate_data = GANDataset(img_dir='data/keras_png_slices_data/keras_png_slices_validate', transform=transform)	
combined_dataset = ConcatDataset([train_data, test_data, validate_data])
data = DataLoader(combined_dataset, batch_size=128, shuffle=True)
