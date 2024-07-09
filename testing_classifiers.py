'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import wandb
wandb.login()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
learning_rate = 0.0001
epochs = 20
# set seed
SEED = 666
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


transform = transforms.Compose([
transforms.ToTensor(),  # convert images to PyTorch tensors
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize the dataset
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, drop_outA="DEFAULT",drop_outB="DEFAULT",drop_outC="DEFAULT"):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.drop_outA = drop_outA
        self.drop_outB = drop_outB
        self.drop_outC = drop_outC
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)
        self.droupout7 = nn.Dropout(p=0.7)
        self.drop_out_dict={
            "0.1":self.dropout1,
            "0.2":self.dropout2,
            "0.3":self.dropout3,
            "0.4":self.dropout4,
            "0.5":self.dropout5,
            "0.6":self.dropout6,
            "0.7":self.droupout7
        }


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.drop_out_dict[self.drop_outA](out)
        out = self.layer2(out)
        out = self.drop_out_dict[self.drop_outB](out)
        out = self.layer3(out)
        out = self.drop_out_dict[self.drop_outC](out)
        out = self.layer4(out)
        out = self.drop_out_dict[self.drop_outA](out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(drop_outA,drop_outB,drop_outC):
    return ResNet(BasicBlock, [2, 2, 2, 2],drop_outA=drop_outA,drop_outB=drop_outB,drop_outC=drop_outC)


def ResNet34(drop_outA,drop_outB,drop_outC):
    return ResNet(BasicBlock, [3, 4, 6, 3],drop_outA=drop_outA,drop_outB=drop_outB,drop_outC=drop_outC)


def ResNet50(drop_outA,drop_outB,drop_outC):
    return ResNet(Bottleneck, [3, 4, 6, 3],drop_outA=drop_outA,drop_outB=drop_outB,drop_outC=drop_outC)


def get_set_accuracy(model, set):
    #also_return y_pred and y_true
    model.eval()
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for images, labels in set:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_pred += predicted.tolist()
            y_true += labels.tolist()


    return correct / total, y_pred, y_true
    


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=epochs):
    model.train()
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):

            images = images.to(device)
            labels = labels.to(device)

            model.train()
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_accuracy,_,_  = get_set_accuracy(model, test_dataloader)
        train_accuracy,_,_ = get_set_accuracy(model, train_dataloader)
        train_losses.append(loss.item())
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)  
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
        wandb.log({"Train Loss": loss.item(), "Train Accuracy": train_accuracy, "Test Accuracy": test_accuracy})
        # add early stopping when overfitting   
        if train_accuracy > 0.92 and  train_accuracy-test_accuracy >0.18 :
            print("Early stopping")
            break
        

    return train_losses, train_accuracies, test_accuracies


def classify():
    wandb.init()
    config = wandb.config
    model_type = config.model
    if model_type == "ResNet50":
        model = ResNet50(config.drop_outsA,config.drop_outsB,drop_outC=config.drop_outsC).to(device)
    elif model_type == "ResNet34":
        model = ResNet34(config.drop_outsA,config.drop_outsB,drop_outC=config.drop_outsC).to(device)
    elif model_type == "ResNet18":
        model = ResNet18(config.drop_outsA,config.drop_outsB,drop_outC=config.drop_outsC).to(device)
    else:
        raise ValueError("Model not supported")


    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_losses, train_accuracies, test_accuracies = train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=epochs)

    # Test the model
    test_accuracy,_,_ = get_set_accuracy(model, test_dataloader)

    print

if __name__ == '__main__':
    print(f'Using device: {device}')
    sweep_config = {
        'method': 'random',
        'parameters': {
            'model': {
                'values': ["ResNet18"]},
            'drop_outsA': {
                'values': ["0.1","0.2","0.3","0.4","0.5","0.6"]
            },
            'drop_outsB': {
                'values': ["0.1","0.2","0.3","0.4","0.5","0.6"]
            },
            'drop_outsC': {
                'values': ["0.1","0.2","0.3","0.4","0.5","0.6"]
            },
        }}
    sweep_id = wandb.sweep(sweep_config, project="Testing_resnets_on_cfar-10")
    wandb.agent(sweep_id, function=classify)

