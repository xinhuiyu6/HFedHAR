import torch
import torch.nn as nn
import torch.nn.functional as F


class EHR_FCN(nn.Module):
    def __init__(self, dim):
        super(EHR_FCN, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(26, 128)  # Adjust the input features to match the output of last conv layer
        self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(48, 16)
        # self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(64, dim)

    def forward(self, x):
        # Assume input shape (batch_size, 900)
        # Reshape to (batch_size, channels, sequence_length)
        bs = x.size(0)
        x = x.view(bs, 26)

        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x1 = x
        x = self.fc5(x)  # Output layer does not use activation as it will be used in CrossEntropyLoss

        return x1, x


