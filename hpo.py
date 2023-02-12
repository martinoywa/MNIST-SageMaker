
# imports
import os
import sys
import logging

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# train function
def train(epochs, model, trainloader, optimizer, criterion, device):
    for i in range(epochs):
        model.train()
        train_loss = 0
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {i}: Train loss: {train_loss:.3f}")

# test function
def test(model, testloader):
    model.to("cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)

    print(f"Test Loss: {test_loss:.4f}, Accuracy: {correct}/{len(testloader.dataset)} ({100.0 * correct / len(testloader.dataset):.0f}%)")

# model class
class Net(nn.ModuleDict):
    def __init__(self):
        super(Net, self).__init__()
        #convolutional layer
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 10)
        # dropout
        self.dropout = nn.Dropout(p=.5)
    
    def forward(self, x):
         # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flatten image input
        x = torch.flatten(x, 1)
        # add hidden layer, with relu activation function
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x
        

# data loader
def dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor()
    ])
    
    trainset = datasets.MNIST("data", train=True, download=True, transform=transform)
    testset = datasets.MNIST("data", train=False, transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size)
    
    return trainloader, testloader

# save model
def save(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

# main function
def main(args):
    # setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"DEVICE: {device}")
    
    # initialize model
    logger.info("INITIALIZE MODEL")
    model = Net()
    model.to(device)
    
    logger.info("CREATING DATA LOADERS")
    trainloader, testloader = dataloaders(args.batch_size)
    
    logger.info("INITIALIZE OPTIMIZER AND CRITERION")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    logger.info("BEGIN TRAINING")
    train(args.epochs, model, trainloader, optimizer, criterion, device)
    
    logger.info("BEGIN TESTING")
    test(model, testloader)
    
    logger.info("SAVE MODEL WEIGHTS")
    save(model, args.model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MNIST HPO")
    parser.add_argument(
        "--batch-size", type=int, default=32, metavar="N", help="input batch size for both training and testing (default:32)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, metavar="N", help="number of epochs to train (default:10)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="N", help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="N", help="momentum for SDG optimizer (default: 0.9)",
    )
    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    
    # Parse
    args = parser.parse_args()
    main(args)
