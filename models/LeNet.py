import torch.nn as nn
import torch.nn.functional as func


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        # self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # self.fc1 = nn.Linear(16*5*5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        # self.conv1 = nn.Conv2d(in_channels=2, out_channels=6, kernel_size=9)
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=6, kernel_size=9)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16*60*60, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(in_features=84, out_features=5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = func.relu(self.conv1(x))  # [batch_size, 6, 248, 248]
        x = func.max_pool2d(x, 2)  # [batch_size, 6, 124, 124]
        x = func.relu(self.conv2(x))  # [batch_size, 16, 120, 120]
        x = func.max_pool2d(x, 2)  # [batch_size, 16, 60, 60]
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))  # [batch_size, 200, 120]
        x = func.relu(self.fc2(x))  # [batch_size, 120, 84]
        x = self.fc3(x)  # [batch_size, 200, 5]
        # x = self.softmax(x) # [batch_size, 200, 5]

        return x