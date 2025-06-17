import torch
import torch.nn as nn
import numpy as np
# import helper
from torch.autograd import Variable
from torch.nn import functional as F
cuda = True if torch.cuda.is_available() else False


class Generator1(nn.Module):

    def __init__(self, n_classes, latent_dim, img_shape):
        # img_shape = [channels, height, width]
        super(Generator1, self).__init__()

        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z, gen_labels):

        # FloatTensor = torch.FloatTensor
        # LongTensor = torch.LongTensor
        # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        # gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

        gen_input = torch.cat((self.label_emb(gen_labels), z), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)

        return img


class Generator2(nn.Module):
    def __init__(self, z_size, input_feat, fc_layers=3, fc_units=400,
                 fc_drop=0, fc_bn=True, fc_n1='relu'):
        super(Generator2, self).__init__()
        self.z_size = z_size
        self.fc_layers = fc_layers
        self.input_feat = input_feat
        self.fc_n1 = fc_n1
        inp_unit = z_size

        # self.layers = nn.ModuleList()
        # self.layers.append(nn.Linear(inp_unit, fc_units, bias=True))
        # for i in range(fc_layers-2):
        #     self.layers.append(nn.Linear(fc_units, fc_units, bias=True))
        # self.layers.append(nn.Linear(fc_units, input_feat, bias=True))

        self.model = nn.Sequential(
            # nn.BatchNorm1d(inp_unit),
            nn.Linear(inp_unit, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, fc_units),
            nn.LeakyReLU(),
            nn.BatchNorm1d(fc_units),

            nn.Linear(fc_units, 2*fc_units),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2*fc_units),

            # nn.Linear(fc_units, fc_units),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(fc_units),

            # nn.Linear(fc_units, fc_units),
            # nn.BatchNorm1d(fc_units),
            # nn.LeakyReLU(),
            nn.Linear(2*fc_units, input_feat),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(input_feat, affine=False)
        )

    def forward(self, z):
        # x = z
        # for i in range(self.fc_layers - 1):
        #     x = F.leaky_relu(self.layers[i](x))
        # x = self.layers[len(self.layers)-1](x)

        x = self.model(z)
        return x


class ConditionalGenerator2(nn.Module):
    def __init__(self, z_size, input_feat, num_classes, fc_layers=3, fc_units=400,
                 fc_drop=0, fc_bn=True, fc_n1='relu'):
        super(ConditionalGenerator2, self).__init__()
        self.z_size = z_size
        self.fc_layers = fc_layers
        self.input_feat = input_feat
        self.fc_n1 = fc_n1
        inp_unit = z_size

        # self.layers = nn.ModuleList()
        # self.layers.append(nn.Linear(inp_unit, fc_units, bias=True))
        # for i in range(fc_layers-2):
        #     self.layers.append(nn.Linear(fc_units, fc_units, bias=True))
        # self.layers.append(nn.Linear(fc_units, input_feat, bias=True))

        self.model = nn.Sequential(
            # nn.BatchNorm1d(inp_unit),
            nn.Linear(inp_unit+num_classes, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, fc_units),
            nn.LeakyReLU(),
            nn.BatchNorm1d(fc_units),

            nn.Linear(fc_units, 2*fc_units),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2*fc_units),

            # nn.Linear(fc_units, fc_units),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(fc_units),

            # nn.Linear(fc_units, fc_units),
            # nn.BatchNorm1d(fc_units),
            # nn.LeakyReLU(),
            nn.Linear(2*fc_units, input_feat),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(input_feat, affine=False)
        )

    def forward(self, x, c):
        # x = z
        # for i in range(self.fc_layers - 1):
        #     x = F.leaky_relu(self.layers[i](x))
        # x = self.layers[len(self.layers)-1](x)

        x = torch.cat((x, c), dim=1)
        x = self.model(x)
        return x


class Generator3(nn.Module):

    def __init__(self,  z_size, input_feat, n_classes=8, fc_units=400):
        super(Generator3, self).__init__()

        input_dim = z_size
        self.label_emb = nn.Embedding(n_classes, n_classes)

        # Original Version
        self.model = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, fc_units),
            nn.BatchNorm1d(fc_units),
            nn.LeakyReLU(),
            nn.Linear(fc_units, fc_units),
            nn.BatchNorm1d(fc_units),
            nn.LeakyReLU(),
            nn.Linear(fc_units, fc_units),
            nn.BatchNorm1d(fc_units),
            nn.LeakyReLU(),
            nn.Linear(fc_units, input_feat),
            nn.Tanh(),
            nn.BatchNorm1d(input_feat, affine=False)
        )
    def forward(self, z, gen_labels):

        # FloatTensor = torch.FloatTensor
        # LongTensor = torch.LongTensor
        # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        # gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

        gen_input = torch.cat((self.label_emb(gen_labels), z), -1)
        x = self.model(gen_input)

        return x


class Generator4(nn.Module):
    def __init__(self, z_size):
        super(Generator4, self).__init__()

        self.z_size = z_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.fc1 = nn.Sequential(nn.Linear(6144, 3000),
                                 nn.ReLU(),
                                 nn.Dropout())
        self.fc2 = nn.Sequential(nn.Linear(3000, 1080),)


    def forward(self, z):
        batch_size = z.shape[0]
        z = z.view(batch_size, 1, 1, self.z_size)
        x = self.conv1(z)
        x = self.conv2(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class BasicConv1d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=False, norm_layer=1):
        super(BasicConv1d, self).__init__()

        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=bias)
        if norm_layer == 1:
            self.bn = nn.BatchNorm1d(out_planes,
                                     eps=0.001,
                                     momentum=0.1,
                                     affine=True)
        elif norm_layer == 2:
            self.bn = nn.InstanceNorm1d(out_planes,
                                        eps=0.001,
                                        momentum=0.1,
                                        affine=True)
        else:
            self.bn = nn.Identity()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Generator_LSTM(nn.Module):

    def __init__(self, hidden_size=256, noise_len=100, output_size=(36, 30), num_layers=2, bidirectional=False):
        super(Generator_LSTM, self).__init__()
        self.output_size = output_size
        self.noise_len = noise_len
        flat_output_size = np.prod(output_size)
        norm_layer = 0
        self.prelstm = nn.Sequential(BasicConv1d(1, 32, 1, norm_layer=norm_layer),
                                     )

        self.lstm = nn.Sequential(nn.LSTM(input_size=noise_len, hidden_size=hidden_size,
                                          num_layers=num_layers, batch_first=True, bidirectional=bidirectional),
                                  )
        self.postlstm = nn.Sequential(BasicConv1d(32, 64, 3, stride=1, padding=1, norm_layer=norm_layer),
                                      BasicConv1d(64, 64, 3, stride=1, padding=1, norm_layer=norm_layer),
                                      )

        self.fcn = nn.Sequential(nn.PReLU(), nn.Dropout(0.2),
                                 nn.Conv1d(64 * 256, flat_output_size, 1),
                                 )
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.normal_(m.weight.data, 0.0, 0.02)
        #     if isinstance(m, nn.BatchNorm1d):
        #         nn.init.normal_(m.weight.data, 1.0, 0.02)
        #         nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        y = self.prelstm(x.view(batch_size, 1, -1))
        y, _ = self.lstm(y.view(batch_size, -1, self.noise_len))
        y = self.postlstm(y)
        y = y.reshape(batch_size, -1, 1)
        y = self.fcn(y)
        y = y.view([batch_size] + list(self.output_size))
        return y


class Discriminator2(nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super(Discriminator2, self).__init__()

        self.linear2 = nn.Linear(hidden_dim, 256)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(256, 64)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear4 = nn.Linear(64, output_dim)
        self.sigmoid = nn.Sigmoid()

        # for m in self.modules():
        #     # print('-----------------------Start initialization------------')
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm1d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, f):
        x = f
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x


class ConditionalDiscriminator2(nn.Module):

    def __init__(self, hidden_dim, output_dim, num_classes):
        super(ConditionalDiscriminator2, self).__init__()

        self.linear2 = nn.Linear(hidden_dim+num_classes, 256)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(256, 64)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear4 = nn.Linear(64, output_dim)
        self.sigmoid = nn.Sigmoid()

        # for m in self.modules():
        #     # print('-----------------------Start initialization------------')
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm1d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, x, c):
        x = torch.cat((x, c), dim=1)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x


class Discriminator3(nn.Module):

    def __init__(self):
        super(Discriminator3, self).__init__()
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

        self.fc3 = nn.Sequential(nn.Linear(512, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.reshape(x.size(0), 3328)
        x = self.fc1(x)
        x = self.fc3(x)
        output = self.sigmoid(x)

        return output


class Discriminator_LSTM(nn.Module):

    def __init__(self, hidden_size=128, input_size=(36, 30), num_layers=2, bidirectional=False):
        super(Discriminator_LSTM, self).__init__()
        self.input_size = input_size
        norm_layer = 0
        self.prelstm = nn.Sequential(BasicConv1d(input_size[0], 64, 3, stride=1, padding=1, norm_layer=norm_layer),
                                     )
        self.lstm = nn.Sequential(nn.LSTM(input_size=input_size[-1], hidden_size=hidden_size,
                                          num_layers=num_layers, batch_first=True, bidirectional=bidirectional),
                                  )

        self.postlstm = nn.Sequential(BasicConv1d(64, 32, 6, stride=3, padding=1, norm_layer=norm_layer),
                                      BasicConv1d(32, 16, 6, stride=3, padding=1, norm_layer=norm_layer),
                                      )

        self.fcn = nn.Sequential(nn.PReLU(), nn.Dropout(0.2),
                                 nn.Conv1d(208, 1, 1),
                                 nn.Sigmoid())
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.normal_(m.weight.data, 0.0, 0.02)
        #     if isinstance(m, nn.BatchNorm1d):
        #         nn.init.normal_(m.weight.data, 1.0, 0.02)
        #         nn.init.constant_(m.bias.data, 0)
        #
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 36, -1)
        y = self.prelstm(x)
        y, _ = self.lstm(y.view(batch_size, -1, self.input_size[-1]))
        y = self.postlstm(y)
        y = y.reshape(batch_size, -1, 1)
        y = self.fcn(y)
        y = y.view(batch_size, 1)
        return y

















