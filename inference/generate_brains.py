import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
# manualSeed = 89127341
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)
# torch.use_deterministic_algorithms(True)

# Get gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# HyperParameters
nz = 256 # Latent feature size
ngf = 128 # Size of feature maps in generator
ndf = 128 # Size of feature maps in discriminator
num_epochs = 20 # Number of training epochs
lr = 0.0002 # Learning rate for optimizers
beta1 = 0.5 # Beta1 hyperparameter for Adam optimizers


# Load data into RAM
dataroot = "data/keras_png_slices_data"

# Number of workers for data loader
# workers = 2
nc = 1 # Only greyscale

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Generator Code
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4

            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)

genner = Generator().to(device)
model = torch.load('./netG_200.pth', weights_only=False)
genner.load_state_dict(model)

# REAL AND FAKE
# Grab a batch of real images from the dataloader
# Select the first `num_images` from the last batch
fixed_noise = torch.randn(128, nz, 1, 1, device=device)
with torch.no_grad():
  fake = genner(fixed_noise).detach().cpu()

selected_images = fake[:16]

# Create a grid of the selected images
grid = vutils.make_grid(selected_images, nrow=4, padding=2, normalize=True)

# single_img.append(fake[0][0].cpu().detach())
             # Display the first image from the fake batch
plt.figure(figsize=(5,5))
plt.imshow(fake[0][0].cpu().detach(), cmap='gray')  # First image, assuming grayscale
plt.gca().set_axis_off()  # Turn off the axis
plt.savefig('./single.png', bbox_inches='tight', pad_inches=0)  # Save without border
plt.show()


# # Plot the grid
plt.figure(figsize=(10,10))
plt.axis("off")
plt.imshow(np.transpose(selected_images, (1, 2, 0)))
plt.savefig('./grid.png', bbox_inches='tight', pad_inches=0)  # Save without border
plt.show()
