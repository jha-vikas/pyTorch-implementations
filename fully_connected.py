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


model = NN(784, 10)
x = torch.randn(64, 784)
assert model(x).shape == torch.Size([64,10])

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size    = 784
num_classes   = 10
learning_rate = 0.001
batch_size    = 64
num_epochs    = 10

# Load Data
train_dataset = datasets.MNIST(root = 'dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader  = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root = 'dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device=device)

# Loss 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda
        data = data.to(device)
        targets = targets.to(device)

        # Get to correct shape
        data = data.reshape(data.shape[0], -1)

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
            x = x.reshape(x.shape[0], -1)
            
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


