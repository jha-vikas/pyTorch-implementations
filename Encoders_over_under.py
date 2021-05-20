import torch
from torch.nn.modules.activation import Tanh                                       # Torch import
import torchvision                                 # importing torchvision
from torch import nn                               # Contains basic building blocks for graphs
from torch.utils.data import DataLoader            # creates iterable/map style over dataset for multiple batches, shuffling
from torchvision import transforms                 # common image transforms which can be chained together using 'Compose'
from torchvision.datasets import MNIST             # MNIST dataset from torchvision
from matplotlib import pyplot as plt


''' torch expects data in form (B, C, H, W) -> Batch(num of image in the batch), Channel, Height, Width

* torch.nn.Functional contains some useful functions like activation functions a convolution operations 
you can use. However, these are not full layers so if you want to specify a layer of any kind you should 
use torch.nn.Module.
* You would use the torch.nn.Functional conv operations to define a custom layer for example with a 
convolution operation, but not to define a standard convolution layer.


Steps:
1. The MNIST iamges are vector of size 784. Images are centered form -1 to 1. To view, centre them at 0.5 with spread between 0 & 1.
2. Display image function: 5 images in a row. use plt.imshow
3. Define dataloader hyperparameters: batch_size, transformations. use torch.nn.Sequential for scripts,, shuffle
4. Specify cuda device
5. Model architecture in class: a. Fully connected AE: Linear+Non-Linear(Tanh). Move to device.
                                b. CNN model: CNN followed by ReLU
                                Keep commented lines to switch between different class of AE
                                define loss criterion: MSELoss (as continuous variable)
6. Define LR. Optimizer
7. Training loop:
    a. forward: apply model, calculate loss
    b. backward
    c. zero_grad, backward, step
'''
# 1. Reshape tensor to proper shape
def to_img(x):
    """Convert vector scaled between -1 & 1, with mean 0 to scaled between 0 & 1, with mean 0.5, 
    followed by putting in size 28x28"""
    x = 0.5 * (x + 1)                             # mean and scaling done
    x = x.view(x.size(0), 28, 28)                 # could have used reshape(but may or may not share storage, view always shares)
    return x

# 2. Display images
def display_images(in_, out, n=1):
    """Take input tensor and display the image from it"""
    for N in range(n):
        if in_ is not None:
            in_pic = to_img(in_.cpu().data)       # move tensor to cpu convert it to 28*28 image format
            plt.figure(figsize=(18, 7.5))         # create a new figure
            for i in range(5):
                plt.subplot(1, 5, i+1)
                plt.imshow(in_pic[i+5*N])
                plt.axis('off')
        out_pic = to_img(out.cpu().data)
        plt.figure(figsize=(18, 7.5))
        for i in range(5):
            plt.subplot(1,5,i+1)
            plt.imshow(out_pic[i+5*N])
            plt.axis('off')




# 3. Data Loading
batch_size = 256

img_transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))
 ])

dataset = MNIST(root= r'../pytorch-Deep-Learning/data/', transform=img_transform,download=True)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# 4. Specifiy device
device = torch.device("cuda:0" if torch.cuda.is_available() else  "cpu")


# 5. Define Model architecture
##n=28x28=784
## For switch between the under over autoencoders
d = 30                                      # for standard AE (under-complete hidden layer)
#d = 500                                    # for denoising AE (over-complete hidden layer)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, d),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d, 28*28),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder().to(device)
criterion = nn.MSELoss()

n_feature = 6

class CNN_autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3,3)),  # from 1 ---> 4, size = 26x26
            #nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(3,3)),  # from 4 --> 16, size = 24x24
            nn.ReLU(),     #
            nn.MaxPool2d(2,2),                                              # size=12x12
            #nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(3,3)),  # 16 --> 4, size=14*14
            #nn.ReLU(),
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=(2,2), stride=2),  # 4 --> 1, size=28*28
            #nn.ReLU(),
            nn.Tanh()                  # TanH is recommended as the output image pixel intensities have to lie between -1 & 1
        )

    def forward(self, x):
        for l in self.encoder:
            x = l(x)
        for l in self.decoder:
            x = l(x)
        return x

conv_model = CNN_autoencoder().to(device)
# criterion remains same

# 6. Optimizer
learning_rate = 1e-3


optimizer = torch.optim.Adam(
    #model.parameters(),       # uncomment for standard AE
    conv_model.parameters(),   # uncomment for CNN AE
    lr=learning_rate
)

# 7. Training loop
num_epochs = 20
do = nn.Dropout()                   # comment out for standard AE
for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img    = img.to(device)
        #img    = img.view(img.size(0), -1)  # tensor to shape mentioned view(). '-1' means other dim is inferred, comment for linear AE

        noise = do(torch.ones(img.shape)).to(device)   # uncomment for variational AE
        #img_bad = (img * noise).to(device)             # uncomment for variational AE

        #************************ forward *************************
        output = conv_model(img)                             # uncomment for fully connected
                                                        # feed img_bad for over AE
        #output = conv_model(img)                       # uncomment for convulation AE
        loss   = criterion(output, img.data)
        #************************ forward *************************
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ***************************** log ***************************
    print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss.item(): .4f}')
    display_images(None, output)                        # (img_bad, output) for variational AE
