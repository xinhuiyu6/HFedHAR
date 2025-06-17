import torch.nn as nn
from torch.nn import functional as F
from models.modules import HSBlock


class CNN_tiny(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(CNN_tiny, self).__init__()

        self.layer1 = self._make_layers(input_channel, 16, (6, 1), (3, 1), (1, 0))
        self.layer2 = self._make_layers(16, 32, (6, 1), (3, 1), (1, 0))
        self.layer3 = self._make_layers(32, 64, (6, 1), (3, 1), (1, 0))
        self.fc = nn.Linear(2304, num_classes)  # 36864

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print('aa',x.shape)
        # x = F.max_pool2d(x, (4,3))
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return x, out


class CNN(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(CNN, self).__init__()

        self.layer1 = self._make_layers(input_channel, 64, (6, 1), (3, 1), (1, 0))
        self.layer2 = self._make_layers(64, 128, (6,1), (3, 1), (1, 0))
        self.layer3 = self._make_layers(128, 256, (6, 1), (3, 1), (1, 0))
        self.fc = nn.Linear(256*4*9, num_classes)

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = F.max_pool2d(x, (4,3))
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        # out = F.softmax(out)
        return x, out


class BasicBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super(BasicBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(output_channel, output_channel, (3,1), 1, (1,0)),
            nn.BatchNorm2d(output_channel),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
        )

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = x + identity
        x = F.relu(x)
        return x


class ResNet_tiny(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(ResNet_tiny, self).__init__()

        self.layer1 = self._make_layers(input_channel, 64, (6,1), (3,1), (1,0))
        self.layer2 = self._make_layers(64, 128, (6,1), (3,1), (1,0))
        self.layer3 = self._make_layers(128, 128, (6,1), (3,1), (1,0))
        self.fc = nn.Linear(4608, num_classes)

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return BasicBlock(input_channel, output_channel, kernel_size, stride, padding)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print('aa',x.shape)
        # x = F.max_pool2d(x, (4,3))
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return x, out


class ResNet_reduced_layers(nn.Module):

    def __init__(self, input_channel, num_classes):
        super(ResNet_reduced_layers, self).__init__()
        self.layer1 = self._make_layers(input_channel, 64, (6,1), (3,1), (1,0))
        self.layer2 = self._make_layers(64, 128, (6,1), (3,1), (1,0))
        # self.layer3 = self._make_layers(128, 128, (6,1), (3,1), (1,0))
        self.fc = nn.Linear(14976, num_classes)

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return BasicBlock(input_channel, output_channel, kernel_size, stride, padding)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # print('aa',x.shape)
        # x = F.max_pool2d(x, (4,3))
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return x, out


class ResNet(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(ResNet, self).__init__()

        self.layer1 = self._make_layers(input_channel, 64, (6, 1), (3, 1), (1, 0))
        self.layer2 = self._make_layers(64, 128, (6, 1), (3, 1), (1, 0))
        self.layer3 = self._make_layers(128, 256, (6, 1), (3, 1), (1, 0))
        self.fc = nn.Linear(256*4*9, num_classes)

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return BasicBlock(input_channel, output_channel, kernel_size, stride, padding)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print('aa',x.shape)
        # x = F.max_pool2d(x, (4,3))
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return x, out


def conv1x1(in_channel, out_channel):
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
                         nn.BatchNorm2d(out_channel)
                         )


def conv3x1(in_channel, out_channel):
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=(8, 1), stride=(2, 1), padding=(0, 0), bias=False),
                         nn.BatchNorm2d(out_channel))


class hs_cnn(nn.Module):

    def __init__(self, out_channel, split, basic_channel):
        super(hs_cnn, self).__init__()

        self.HS_conv = HSBlock(out_channel, split, basic_channel)

        if out_channel % split == 0:
            self.last_conv = conv1x1((out_channel//split + basic_channel + (split-2)*(basic_channel//2)), out_channel)
        else:
            self.last_conv = conv1x1((out_channel//split + 1 + basic_channel + (split-2)*(basic_channel//2)), out_channel)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.HS_conv(x)
        out = self.last_conv(out)
        out = self.relu(out)
        return out


class HS_CNN(nn.Module):
    def __init__(self):
        super(HS_CNN, self).__init__()
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            hs_cnn(64, 4, 28)
        )
        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            hs_cnn(128, 4, 28)
        )
        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            hs_cnn(256, 4, 28)
        )
        self.fc = nn.Sequential(
            nn.Linear(9216, 6)
        )

    def forward(self, x):
        # print(x.shape)
        out = self.Block1(x)
        out = self.Block2(out)
        out = self.Block3(out)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        # print(out.shape)
        out = nn.LayerNorm(out.size())(out.cpu())
        # print(out.shape)
        out = out.cuda()
        return out, out


class ActivityLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=6, num_layers=1):
        super(ActivityLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x, hidden1 = self.lstm1(x)
        x, hidden2 = self.lstm2(x)
        # x, hidden3 = self.lstm3(x)
        # x = self.dropout(x)
        out = x[:, -1, :]  # Take the last time step output
        out = self.fc(out)
        return x, out