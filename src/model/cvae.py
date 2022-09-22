import torch
import torch.nn as nn


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()

        self.X_dim = 7
        self.Y_dim = 3
        self.input_dim = self.X_dim + self.Y_dim
        self.z_dim = 100
        self.mean = nn.Linear(200, self.z_dim)
        self.logvar = nn.Linear(200, self.z_dim)

        self.Encoder = [
            nn.Linear(self.input_dim, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
        ]

        self.Decoder = [
            nn.Linear(self.z_dim + self.Y_dim, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(),
            nn.Linear(200, self.X_dim),
            nn.BatchNorm1d(self.X_dim),
            nn.Softmax(dim=1),
        ]

        self.Encoder = nn.Sequential(*self.Encoder)
        self.Decoder = nn.Sequential(*self.Decoder)

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def get_normal_distribution_param(self, encoded_xy):
        mean = self.mean(encoded_xy)
        logvar = self.logvar(encoded_xy)
        return mean, logvar

    def forward(self, x, y):

        xy = torch.cat((x, y), axis=1)
        encoded_xy = self.Encoder(xy)

        mean, logvar = self.get_normal_distribution_param(encoded_xy)

        z = self.reparameterize(mean, logvar)
        zy = torch.cat((z, y), axis=1)
        decoded_zy = self.Decoder(zy)

        return z, mean, logvar, decoded_zy
