# imports
import torch
import torch.nn as nn
import torch.optim as optim     # optimiser
import torch.nn.functional as F # has all functions which don't have any parameters (Eg: ReLU)
from torch.utils.data import DataLoader # Dataset management
import torchvision.datasets as datasets # standard datasets
import torchvision.transforms as transforms

# Create RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.rnn         = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc          = nn.Linear(hidden_size*sequence_length, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)


        # forward pass
        out, _ = self.rnn(x, h0)
        out    = out.reshape(out.shape[0], -1)
        out    = self.fc(out)
        return out


# Create RNN
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.gru         = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc          = nn.Linear(hidden_size*sequence_length, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)


        # forward pass
        out, _ = self.gru(x, h0)
        out    = out.reshape(out.shape[0], -1)
        out    = self.fc(out)
        return out


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size    = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes   = 10
learning_rate = 0.001
batch_size    = 64
num_epochs    = 2




# Load Data
train_dataset = datasets.MNIST(root = 'dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader  = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root = 'dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize network
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

model1 = GRU(input_size, hidden_size, num_layers, num_classes).to(device)
# Loss 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda
        data = data.to(device).squeeze(1)
        targets = targets.to(device)

        # Get to correct shape
        # data = data.reshape(data.shape[0], -1)

        # forward
        scores = model1(data)
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
    model1.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).squeeze(1)
            y = y.to(device)
            #x = x.reshape(x.shape[0], -1)
            
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        
        acc = float(num_correct)/float(num_samples)*100
        print(f'Accuracy: {acc}')

    model1.train()
    return acc

check_accuracy(train_loader, model1)
check_accuracy(test_loader, model1)


