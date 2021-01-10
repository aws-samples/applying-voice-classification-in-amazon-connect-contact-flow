"""

# Reference papers

- [Very Deep Convolutional Neural Networks for Raw Waveforms]
  (https://arxiv.org/abs/1610.00087)

"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from coswara_dataset import CoswareDataset
from pathlib import Path


class NetM3(nn.Module):
    def __init__(self, num_classes=2):
        super(NetM3, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(30)  # input should be 512x30 so this outputs a 512x1
        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1)  # change the 512x1 to 1x512
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)  # Output: torch.Size([N, 1, num_classes])


if __name__ == "__main__":
    model = NetM3(7)

    print("Test 1:")
    data = torch.ones([1, 1, 32000])
    print("input:", data.shape)
    output = model(data)
    print("output:", output.shape)

    print("Test 2:")
    datapath = Path("/home/ec2-user/SageMaker/Coswara-Data")
    csvpath = datapath / "breathing-deep-metadata.csv"
    dataset = CoswareDataset(
        csv_path=csvpath,
        file_path=datapath,
        new_sr=8000,
        audio_len=20,
        sampling_ratio=5,
    )
    # Check 1st samples
    data, label = dataset[0]
    data = data.unsqueeze(0)
    print("input:", data.shape)
    output = model(data)
    print("output:", output.shape)
