# 1. Define libraries
import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


# 2. Specifiy device
device = torch.device("cuda:0" if torch.cuda.is_available() else  "cpu")

#img_dim for MNIST is 28x28x1 = 784

# 3. Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1), # LeakyReLU is better for GANs for gradient transfer during backpropogation
            nn.Linear(128, 1),  # Gives output value
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.disc(x)

# 4. Generator    
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # output of pixel values are between -1 & 1 post normalization
        )

    def forward(self, x):
        return self.gen(x)

# 5. Hyperparameters:
## GANs are very sensitive to hyperparameters. These hyperparameters are using
## the actual GAN paper hyperparameters
lr = 3e-4 # from Andrej Karpathy and Aladdin Persson
z_dim = 64 # latent space dim
image_dim = 25*28*1 # 784
batch_size = 32
num_epochs = 50

disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)  # to check how it changes acriss epochs
