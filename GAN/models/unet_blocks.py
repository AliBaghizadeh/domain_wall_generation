# Import Libraries
import torch
import torch.nn as nn
from domain_wall_generation.gan.models.film import FiLM, LatentFiLM

# Import Finished

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, n_classes, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 4, 2, 1)
        self.bn = nn.BatchNorm2d(out_size)
        self.film = FiLM(out_size, n_classes)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, skip_input, labels):
        x = self.up(x)
        x = self.bn(x)
        x = self.film(x, labels)
        x = self.activation(x)
        x = self.dropout(x)
        return torch.cat((x, skip_input), 1)

class UNetUp_latent(nn.Module):
    def __init__(self, in_size, out_size, n_classes, z_dim=8, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 4, 2, 1)
        self.bn = nn.BatchNorm2d(out_size)
        self.film_label = FiLM(out_size, n_classes)
        self.film_latent = LatentFiLM(z_dim, out_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, skip_input, labels, z):
        x = self.up(x)
        x = self.bn(x)
        x = self.film_label(x, labels)
        x = self.film_latent(x, z)
        x = self.activation(x)
        x = self.dropout(x)
        return torch.cat((x, skip_input), 1)

# Inject z into the encoder (UNetDown) as early conditioning and deeper into network

class UNetDown_latent(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, z_dim=None):
        super().__init__()
        self.use_latent = z_dim is not None
        self.conv = nn.Conv2d(in_size, out_size, 4, 2, 1)
        self.norm = nn.BatchNorm2d(out_size) if normalize else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.film_z = LatentFiLM(z_dim, out_size) if self.use_latent else nn.Identity()

    def forward(self, x, z=None):
        x = self.conv(x)
        x = self.norm(x)
        if self.use_latent:
            x = self.film_z(x, z)
        x = self.act(x)
        return x