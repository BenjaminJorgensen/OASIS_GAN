#!/usr/bin/env python3
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Seed is Student-Number
manualSeed = 4717300
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)

# HyperParameters
latent_size = 256 # Latent feature size
n_generator_features = 128 # Size of feature maps in generator
n_descriminator_features = 128 # Size of feature maps in discriminator
num_epochs = 200 # Number of training epochs
learning_rate = 0.0002 # Learning rate for optimizers
beta1 = 0.5 # Beta1 hyperparameter for Adam optimizers

# Image Information
image_dir = "data/keras_png_slices_data"
channels = 1 # Only greyscale
image_size = 128
img_shape = (channels, image_size, image_size)

# Load in Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
     transforms.Resize(image_size),
     transforms.CenterCrop(image_size),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load all images into a data loader
image_data = datasets.ImageFolder(root=image_dir, transform=transform)
dataloader = DataLoader(image_data, batch_size=256, shuffle=True)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Initialise Weights - Normal Distribution with mean 1 and std 0.02
# Initialising weights as Gaussian's helps models converge faster
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Goal of generator is to make images from latent layer
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(

            # Start with the latent features as input
            nn.ConvTranspose2d(latent_size, n_generator_features * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_generator_features * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(n_generator_features * 16, n_generator_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_generator_features * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(n_generator_features * 8, n_generator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_generator_features * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(n_generator_features * 4, n_generator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_generator_features * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(n_generator_features * 2, n_generator_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_generator_features),
            nn.ReLU(True),

            # End with a single chanell with 2D image
            nn.ConvTranspose2d(n_generator_features, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Initialising the Generator
netG = Generator().to(device)
netG.apply(weights_init)

# Goal of the descriminator is to determine if an image is real or generated
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
                
            nn.Conv2d(channels, n_descriminator_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_descriminator_features, n_descriminator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_descriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_descriminator_features * 2, n_descriminator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_descriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_descriminator_features * 4, n_descriminator_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_descriminator_features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_descriminator_features * 8, n_descriminator_features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_descriminator_features * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_descriminator_features * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

netD = Discriminator().to(device)
netD.apply(weights_init)

# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(128, latent_size, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate/9, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Training Loop
# Lists to keep track of progress
img_list = []
single_img = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        # Training descriminent - Real data
        # maximize log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()
        real_batch = data[0].to(device)
        b_size = real_batch.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_batch).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        ## Train Discriminant with all-fake data
        noise = torch.randn(b_size, latent_size, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # Train Generator: Maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 100 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            single_img.append(fake[0][0].cpu().detach())

            # Display the image every 100 iterations to view progression
            plt.figure(figsize=(5,5))
            plt.axis("off")
            plt.imshow(fake[0][0].cpu().detach(), cmap='gray')
            plt.show()
        iters += 1


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('./Loss.png', bbox_inches='tight')
plt.show()

# Plot fake images
images_to_show = img_list[-1][:9]
# Create a grid with 3x3 images
grid = vutils.make_grid(images_to_show, nrow=3, padding=2, normalize=True)

# Plot the grid of images
plt.figure(figsize=(10,10))  # Adjust size if needed
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(grid.cpu(), (1,2,0)))
plt.savefig('./fake_brains.png', bbox_inches='tight')
plt.show()


