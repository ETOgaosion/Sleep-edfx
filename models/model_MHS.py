import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class EMDNet(nn.Module):
    def __init__(self):
        super(EMDNet, self).__init__()
        self.T = 3000

        # Conv Layer 1
        self.conv1 = nn.Conv1d(8, 64, 100, 12, padding=0)
        self.batchnorm1 = nn.BatchNorm1d(64, False)

        # Pooling Layer 1
        self.pooling1 = nn.MaxPool1d(8, 8)

        # Conv Layer 2
        self.conv2 = nn.Conv1d(64, 128, 8, 1, padding=0)
        self.batchnorm2 = nn.BatchNorm1d(128, False)

        # Conv Layer 3
        self.conv3 = nn.Conv1d(128, 128, 8, 1, padding=0)
        self.batchnorm3 = nn.BatchNorm1d(128, False)

        # Conv Layer 4
        self.conv4 = nn.Conv1d(128, 128, 8, 1, padding=0)
        self.batchnorm4 = nn.BatchNorm1d(128, False)

        # Pooling Layer 2
        self.pooling2 = nn.MaxPool1d(4, 4)

        # Fully Connection Layer
        self.fc1 = nn.Linear(128*2, 5)

    def forward(self, x):
        # Conv Layer 1
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)

        # Pooling Layer 1
        x = self.pooling1(x)
        x = F.dropout(x, 0.5)

        # Conv Layer 2
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)

        # Conv Layer 3
        x = F.relu(self.conv3(x))
        x = self.batchnorm3(x)

        # Conv Layer 4
        x = F.relu(self.conv3(x))
        x = self.batchnorm4(x)

        # Pooling Layer 2
        x = self.pooling2(x)
        x = F.dropout(x, 0.5)

        # Fully Connection Layer
        x = x.view(-1, 128*2)
        x = F.softmax(self.fc1(x), dim=1)

        return x


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 3000

        # Conv Layer 1
        self.conv1 = nn.Conv1d(1, 64, 100, 12, padding=0)
        self.batchnorm1 = nn.BatchNorm1d(64, False)

        # Pooling Layer 1
        self.pooling1 = nn.MaxPool1d(8, 8)

        # Conv Layer 2
        self.conv2 = nn.Conv1d(64, 128, 8, 1, padding=0)
        self.batchnorm2 = nn.BatchNorm1d(128, False)

        # Conv Layer 3
        self.conv3 = nn.Conv1d(128, 128, 8, 1, padding=0)
        self.batchnorm3 = nn.BatchNorm1d(128, False)

        # Conv Layer 4
        self.conv4 = nn.Conv1d(128, 128, 8, 1, padding=0)
        self.batchnorm4 = nn.BatchNorm1d(128, False)

        # Pooling Layer 2
        self.pooling2 = nn.MaxPool1d(4, 4)

        # Fully Connection Layer
        self.fc1 = nn.Linear(128 * 2, 5)

    def forward(self, x):
        # Conv Layer 1
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)

        # Pooling Layer 1
        x = self.pooling1(x)
        x = F.dropout(x, 0.5)

        # Conv Layer 2
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)

        # Conv Layer 3
        x = F.relu(self.conv3(x))
        x = self.batchnorm3(x)

        # Conv Layer 4
        x = F.relu(self.conv3(x))
        x = self.batchnorm4(x)

        # Pooling Layer 2
        x = self.pooling2(x)
        x = F.dropout(x, 0.5)

        # Fully Connection Layer
        x = x.view(-1, 128 * 2)
        x = F.softmax(self.fc1(x), dim=1)

        return x


class MHSNet(nn.Module):
    def __init__(self):
        super(MHSNet, self).__init__()
        self.T = 2999

        # Conv Layer 1
        self.conv1 = nn.Conv1d(8, 64, 100, 12, padding=0)
        self.batchnorm1 = nn.BatchNorm1d(64, False)

        # Pooling Layer 1
        self.pooling1 = nn.MaxPool1d(8, 8)

        # Conv Layer 2
        self.conv2 = nn.Conv1d(64, 128, 8, 1, padding=0)
        self.batchnorm2 = nn.BatchNorm1d(128, False)

        # Conv Layer 3
        self.conv3 = nn.Conv1d(128, 128, 8, 1, padding=0)
        self.batchnorm3 = nn.BatchNorm1d(128, False)

        # Conv Layer 4
        self.conv4 = nn.Conv1d(128, 128, 8, 1, padding=0)
        self.batchnorm4 = nn.BatchNorm1d(128, False)

        # Pooling Layer 2
        self.pooling2 = nn.MaxPool1d(4, 4)

        # Fully Connection Layer
        self.fc1 = nn.Linear(128*2, 5)

    def forward(self, x):
        # Conv Layer 1
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)

        # Pooling Layer 1
        x = self.pooling1(x)
        x = F.dropout(x, 0.5)

        # Conv Layer 2
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)

        # Conv Layer 3
        x = F.relu(self.conv3(x))
        x = self.batchnorm3(x)

        # Conv Layer 4
        x = F.relu(self.conv3(x))
        x = self.batchnorm4(x)

        # Pooling Layer 2
        x = self.pooling2(x)
        x = F.dropout(x, 0.5)

        # Fully Connection Layer
        x = x.view(-1, 128*2)
        x = F.softmax(self.fc1(x), dim=1)

        return x
