import torch
import torch.nn as nn


class Generator2(nn.Module):
    def __init__(self, z_size, input_feat, fc_layers=3, fc_units=400,
                 fc_drop=0, fc_bn=True, fc_n1='relu'):
        super(Generator2, self).__init__()

        self.z_size = z_size
        self.fc_layers = fc_layers
        self.input_feat = input_feat
        self.fc_n1 = fc_n1
        inp_unit = z_size

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(inp_unit, fc_units, bias=True))
        for i in range(fc_layers-2):
            self.layers.append(nn.Linear(fc_units, fc_units, bias=True))
        self.layers.append(nn.Linear(fc_units, input_feat, bias=True))

        self.model = nn.Sequential(

            nn.Linear(inp_unit, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, fc_units),
            nn.LeakyReLU(),
            nn.BatchNorm1d(fc_units),

            nn.Linear(fc_units, fc_units),  # fc_units*3
            nn.LeakyReLU(),
            nn.BatchNorm1d(fc_units),  # fc_units*3

            nn.Linear(fc_units, input_feat),  # 3
        )

    def forward(self, z):

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

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(inp_unit, fc_units, bias=True))
        for i in range(fc_layers-2):
            self.layers.append(nn.Linear(fc_units, fc_units, bias=True))
        self.layers.append(nn.Linear(fc_units, input_feat, bias=True))

        self.model = nn.Sequential(

            nn.Linear(inp_unit+num_classes, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, fc_units),
            nn.LeakyReLU(),
            nn.BatchNorm1d(fc_units),

            nn.Linear(fc_units, fc_units),  # fc_units*3
            nn.LeakyReLU(),
            nn.BatchNorm1d(fc_units),  # fc_units*3

            nn.Linear(fc_units, input_feat),  # 3
        )

    def forward(self, z, c):
        x = torch.cat((z, c), dim=1)

        x = self.model(x)
        return x


class Discriminator2(nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super(Discriminator2, self).__init__()

        self.linear2 = nn.Linear(hidden_dim, 64)  # 256
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(64, 64)  # 256
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

        self.linear2 = nn.Linear(hidden_dim+num_classes, 64)  # 256
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(64, 64)  # 256
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