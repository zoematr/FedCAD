"""FLBreast: A Flower / PyTorch app for BreastMNIST."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize, ToTensor
from medmnist import BreastMNIST
import numpy as np

# ---------------------------
# Model definition
# ---------------------------
class Net(nn.Module):
    """Simple CNN for grayscale 28×28 images (BreastMNIST)."""

    def __init__(self):
        super(Net, self).__init__()
        # 1 input channel (grayscale), 2 output classes
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 2)  # binary classification (2 classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------
# Data loading
# ---------------------------
def load_data(partition_id: int, num_partitions: int):
    """Load partitioned BreastMNIST data for federated training."""

    # Define transforms (normalize grayscale images)
    transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])

    # Download dataset (train split only)
    full_train = BreastMNIST(split="train", download=True, transform=transform)
    total_size = len(full_train)
    indices = np.arange(total_size)

    # Shuffle indices deterministically
    rng = np.random.default_rng(seed=42)
    rng.shuffle(indices)

    # Partition the dataset into roughly equal shards
    partition_size = total_size // num_partitions
    start = partition_id * partition_size
    end = start + partition_size if partition_id < num_partitions - 1 else total_size
    part_indices = indices[start:end]

    # Create subset for this client
    client_subset = Subset(full_train, part_indices)

    # Local 80/20 train/test split
    local_size = len(client_subset)
    split = int(0.8 * local_size)
    local_indices = np.arange(local_size)
    rng.shuffle(local_indices)
    train_idx, test_idx = local_indices[:split], local_indices[split:]

    train_subset = Subset(client_subset, train_idx)
    test_subset = Subset(client_subset, test_idx)

    # DataLoaders
    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=32, shuffle=False)

    return trainloader, testloader


# ---------------------------
# Training and testing loops
# ---------------------------
def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0

    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
