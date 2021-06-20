# 1. Define libraries
from ctypes import create_unicode_buffer
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
image_dim = 28*28*1 # 784
batch_size = 32
num_epochs = 50

# 6. Defining the model
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)  # to check how it changes acriss epochs

# 7. Dataset download and transformation
transforms = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))]
)

dataset = datasets.MNIST(root='dataset/', transform=transforms, download=True)

loader = DataLoader(dataset, batch_size, shuffle=True)

# 8. Optimizer
opt_disc = optim.Adam(disc.parameters(), lr)
opt_gen = optim.Adam(gen.parameters(), lr)
criterion = nn.BCELoss()

# 9. Tensorboard
writer_fake = SummaryWriter(f'GAN_MNIST/fake')
writer_real = SummaryWriter(f'GAN_MNIST/real')
step = 0

# 10. Loop
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]   #The actual batch_size(can be different from batch size defined above for last batch)

        ## Train Discriminator: max y_real*log(D(real)) + (1-y_real)log(1-D(G(z))) (Based on paper which intrdouced GAN)
        ## y_real will be 1 for real, and 0 for fake
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake.detach()).view(-1)
        # we detach so as not to clear the intermediate computation when 
        # backward pass is run as we need them for training gen
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward()  # or lossD.backward(retain_graph=True) instead of detaching the disc_fake
        opt_disc.step()

        ## For training the generator, min log(1 - D(G(z))
        ## thats equivalent to max(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()


        # Code for tensorboard
        if batch_idx == 0:
            print(
                f"Epoch [{epoch} / {num_epochs}] \ "
                f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(fake, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )

                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )

                step += 1
