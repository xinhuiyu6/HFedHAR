import torch.nn as nn
from torch.nn import functional as F
from models.modules import _CrossNeuronBlock, HSBlock, CE
import torch

class FC_model_parent(nn.Module):

    def __init__(self, classes):
        super(FC_model_parent, self).__init__()
        self.classes = classes
        self.layer1 = nn.Sequential(
            nn.Linear(1080, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        # self.layer4 = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2)
        # )
        self.output = nn.Linear(128, self.classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        logits = self.output(x)
        return x, logits


class FC_model_parent_parent(nn.Module):

    def __init__(self, classes):
        super(FC_model_parent_parent, self).__init__()
        self.classes = classes
        self. layer1 = nn.Sequential(
            nn.Linear(1080, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.output = nn.Linear(64, self.classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        logits = self.output(x)
        return x, logits


class FC_model(nn.Module):

    def __init__(self, classes):
        super(FC_model, self).__init__()
        self.classes = classes
        self.layer1 = nn.Sequential(
            nn.Linear(1080, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.output = nn.Linear(256, self.classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        feats = x
        logits = self.output(x)
        return feats, logits


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=27, out_channels=32, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.Dropout(p=0.1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.Dropout(p=0.1)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # nn.Dropout(p=0.1)
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(1152, 12)
        # )


    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


class Encoder_1d(nn.Module):

    def __init__(self):
        super(Encoder_1d, self).__init__()
        self. layer1 = nn.Sequential(
            nn.Linear(1080, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class Encoder_1d_1(nn.Module):

    def __init__(self):
        super(Encoder_1d_1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(1080, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Decoder_1d(nn.Module):

    def __init__(self, encoder):
        super(Decoder_1d, self).__init__()

        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1080),
            # nn.ReLU(inplace=True),
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class sslmodel_1d(nn.Module):

    def __init__(self, encoder):
        super(sslmodel_1d, self).__init__()

        self.encoder = encoder
        self.proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.proj(x)
        return x


class Linear_model_1d(nn.Module):
    def __init__(self, encoder, in_size, out_size):
        super(Linear_model_1d, self).__init__()

        self.encoder = encoder
        self.linear = nn.Linear(in_size, out_size)
        # self.output = nn.Linear(32, out_size)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.encoder(x)
        feats = x
        x = self.linear(x)
        # x = self.output(x)
        return feats, x


class Projection(nn.Module):
    def __init__(self, in_size, hidden=[256, 128]):
        super(Projection).__init__()

        self.relu = nn.ReLU()
        self.linear1 = nn.Sequential(
            nn.Linear(in_size, hidden[0]),
            nn.BatchNorm1d(hidden[0]),
            nn.ReLU(inplace=True)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1])
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)

        return x


class Linear_model(nn.Module):
    def __init__(self, in_size, out_size):
        super(Linear_model, self).__init__()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x


class sslmodel(nn.Module):

    def __init__(self, encoder):
        super(sslmodel, self).__init__()

        self.encoder = encoder
        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = nn.Flatten()(x)
        x = self.proj(x)
        return x


class slmodel(nn.Module):
    def __init__(self, in_size, out_size):
        super(slmodel).__init__()

        self.encoder = Encoder()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):

        x = self.encoder(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class conv_net(nn.Module):
    def __init__(self):
        super(conv_net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.fc1 = nn.Sequential(nn.Linear(3328, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2))

        self.fc3 = nn.Sequential(nn.Linear(512, 8))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # print(x.size(0))

        x = x.reshape(x.size(0), 3328)
        x = self.fc1(x)
        output = self.fc3(x)

        return x, output


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = self._make_layers(36, 64, (1, 3), (1, 1), 0)
        self.layer2 = self._make_layers(64, 128, (1, 3), (1, 1), 0)
        self.layer3 = self._make_layers(128, 256, (1, 3), (1, 1), 0)
        self.fc = nn.Linear(256*24*1, 8)

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
        x1 = x
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return x1, out


class Net_SC(nn.Module):
    def __init__(self):
            super(Net_SC, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )

            # self.C2 = nn.Sequential(
            #     _CrossNeuronBlock(64, 1, 28, spatial_height=8, spatial_width=8, reduction=12),
            #     nn.BatchNorm2d(64),
            #     nn.ReLU(True)
            # )
            # self.layer3 = nn.Sequential(
            #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
            #     nn.BatchNorm2d(128),
            #     nn.ReLU(True)
            # )

            self.C2 = nn.Sequential(
                _CrossNeuronBlock(64, 28, 34, spatial_height=8, spatial_width=8, reduction=12),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
            )

            self.fc = nn.Sequential(
                nn.Linear(106496, 8)
            )

    def forward(self, x):
        x = self.layer1(x)
        # x = self.layer2(x)
        x = self.C2(x)
        x = self.layer3(x)

        # x = self.layer1(x)
        # x = self.C2(x)
        # x = self.layer3(x)

        x = x.view(x.size(0), -1)
        x1 = x
        x = self.fc(x)
        # x = F.normalize(x.cuda())
        return x1, x



class HConv(nn.Module):
    def __init__(self, input_channel, kernel_size=(5,1), stride=1, padding=(1,0), pooling_r=(5,1)):
        super(HConv, self).__init__()

        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(input_channel, input_channel, kernel_size, stride, padding),
                    nn.BatchNorm2d(input_channel),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(input_channel, input_channel, kernel_size, stride, padding),
                    nn.BatchNorm2d(input_channel),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(input_channel, input_channel, kernel_size, stride, padding),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU(inplace=True),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(F.interpolate(self.k2(x), identity.size()[2:]))
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out

#----------CNN----------
class HCBlock(nn.Module):
    def __init__(self, input_channel, kernel_size, stride, padding):
        super(HCBlock, self).__init__()

        self.HC = HConv(input_channel//2, kernel_size, stride, padding)
        self.k1 = nn.Sequential(
            nn.Conv2d(input_channel//2, input_channel//2, kernel_size, stride, padding),
            nn.BatchNorm2d(input_channel//2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out_a, out_b = torch.split(x, [x.size(1)//2, x.size(1)//2], dim=1)
        out_a = self.k1(out_a)
        out_b = self.HC(out_b)
        out = torch.cat([out_a, out_b], dim=1)

        return out


class CNN_HC(nn.Module):
    def __init__(self):
        super(CNN_HC, self).__init__()

        self.layer1 = self._make_layers(1, 64, (3, 1), (1, 1), (1, 0))
        self.layer2 = self._make_layers(64, 64, (3, 1), (1, 1), (1, 0), HC=True)
        self.layer3 = self._make_layers(64, 128, (3, 1), (1, 1), (1, 0))
        self.fc = nn.Linear(138240, 8)

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding, HC=False):
        if HC == True:
            return HCBlock(input_channel, kernel_size, stride, padding)
        else:
            return nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channel),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x1 = x
        out = self.fc(x)

        return x1, out


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

        # self.first_conv = conv3x3(in_channel, out_channel)

        self.HS_conv = HSBlock(out_channel, split, basic_channel)

        if out_channel % split == 0:
            self.last_conv = conv1x1((out_channel//split + basic_channel + (split-2)*(basic_channel//2)), out_channel)
        else:
            self.last_conv = conv1x1((out_channel//split + 1 + basic_channel + (split-2)*(basic_channel//2)), out_channel)

        # self.shortcut_conv = conv3x3(in_channel, out_channel)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # out = self.first_conv(x)
        # print(out.shape, 'out1')
        out = self.HS_conv(x)
        # print(out.shape, 'out2')
        out = self.last_conv(out)
        # y = self.shortcut_conv(x)
        # out += y
        out = self.relu(out)
        return out


class HS_CNN(nn.Module):
    def __init__(self):
        super(HS_CNN, self).__init__()
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 1), stride=(2, 1), padding=(1,0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            hs_cnn(64, 4, 28)
        )
        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1,0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            hs_cnn(128, 4, 28)
        )
        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            hs_cnn(256, 4, 28)
        )
        self.fc = nn.Sequential(
            nn.Linear(27648, 8)
        )

    def forward(self, x):
        # print(x.shape)
        out = self.Block1(x)
        out = self.Block2(out)
        out = self.Block3(out)
        out = out.view(out.size(0), -1)
        x1 = out
        # print(out.shape)
        out = self.fc(out)
        # print(out.shape)
        out = nn.LayerNorm(out.size())(out.cpu())
        # print(out.shape)
        out = out.cuda()
        return x1, out


class DeepConvLSTM(nn.Module):
    def __init__(self, n_channels=1, n_classes=8, conv_kernels=64, kernel_size=4, LSTM_units=128, backbone=False):
        super(DeepConvLSTM, self).__init__()

        self.backbone = backbone

        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1), stride=(2, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels*2, (kernel_size, 1), stride=(2, 1), padding=(1, 0))

        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(4608, LSTM_units, num_layers=1)

        self.out_dim = LSTM_units

        if backbone == False:
            self.classifier = nn.Linear(LSTM_units, n_classes)

        self.BN1 = nn.BatchNorm2d(64)
        self.BN2 = nn.BatchNorm2d(128)

        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()


    def forward(self, x):
        self.lstm.flatten_parameters()

        x = self.activation1(self.BN1(self.conv1(x)))
        x = self.activation2(self.BN2(self.conv2(x)))

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)

        x, h = self.lstm(x)
        x = x[-1, :, :]

        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return x, out


class HS_CNN_LSTM(nn.Module):
    def __init__(self):
        super(HS_CNN_LSTM, self).__init__()
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 1), stride=(2, 1), padding=(1,0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            hs_cnn(64, 4, 28)
        )
        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1,0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            hs_cnn(128, 4, 28)
        )
        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            hs_cnn(256, 4, 28)
        )

        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(9216, 128, num_layers=1)

        self.fc = nn.Sequential(
            nn.Linear(128, 8)
        )

    def forward(self, x):
        # print(x.shape)
        out = self.Block1(x)
        out = self.Block2(out)
        x = self.Block3(out)

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)

        x, h = self.lstm(x)
        x = x[-1, :, :]

        out = self.fc(x)
        return x, out