# imports
import torch
import torch.nn as nn
import torch.optim as optim     # optimiser
import torch.nn.functional as F # has all functions which don't have any parameters (Eg: ReLU)
from torch.utils.data import DataLoader # Dataset management
import torchvision.datasets as datasets # standard datasets
import torchvision.transforms as transforms

# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Create CNN class
class CNN(nn.Module):
    def __init__(self, in_channel = 1, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool  = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1   = nn.Linear(16*7*7, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


model = CNN()
x = torch.randn(64, 1, 28, 28)
print(model(x).shape)


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channel    = 1
input_size    = 784
num_classes   = 10
learning_rate = 0.002
batch_size    = 64
num_epochs    = 5

# Load Data
train_dataset = datasets.MNIST(root = 'dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader  = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root = 'dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize network
model = CNN().to(device=device)

# Loss 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda
        data = data.to(device)
        targets = targets.to(device)

        # forward
        scores = model(data)
        loss   = criterion(scores, targets)

        # backward
        # it is implemented in pyTorch
        optimizer.zero_grad()  # to reset all grads to 0 so as it doesn't have values from previous step
        loss.backward()

        # backpropogation and GD
        optimizer.step()


# Check accuracy on train and test
def check_accuracy(loader: DataLoader, model: model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        
        acc = float(num_correct)/float(num_samples)*100
        print(f'Accuracy: {acc}')

    model.train()
    return acc

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

