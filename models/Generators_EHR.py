import torch.nn as nn


class Generator2(nn.Module):
    def __init__(self, z_size, fc_units=128, input_feat=26):
        super(Generator2, self).__init__()

        self.z_size = z_size
        inp_unit = z_size

        self.model = nn.Sequential(
            nn.Linear(inp_unit, fc_units),
            nn.LeakyReLU(),
            nn.BatchNorm1d(fc_units),
            # nn.LayerNorm(fc_units),

            nn.Linear(fc_units, 2*fc_units),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2*fc_units),
            # nn.LayerNorm(2*fc_units),

            nn.Linear(fc_units*2, input_feat),  # 3
            nn.LeakyReLU()
        )

    def forward(self, z):
        x = self.model(z)
        return x


class Discriminator2(nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super(Discriminator2, self).__init__()

        self.linear2 = nn.Linear(hidden_dim, 128)  # 256
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(128, 64)  # 256
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