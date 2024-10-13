import torch
from torch import nn
from torch.nn import Module

from models.stylegan2.model import EqualLinear, PixelNorm

STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]

class MapperBase(Module):

    def __init__(self, latent_dim=512):
        super(MapperBase, self).__init__()

        layers = [PixelNorm()]

        for i in range(4):
            layers.append(
                EqualLinear(
                    latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                )
            )

        self.mapping = nn.Sequential(*layers)


    def forward(self, x):
        x = self.mapping(x)
        return x


class UpScaleEmbed(Module):
    def __init__(self,embed_dim=512, #adaptive
                 latent_dim=512, n_layer=18):
        super(UpScaleEmbed, self).__init__()
        self.linear = EqualLinear(embed_dim, n_layer * latent_dim, lr_mul=1)
        self.n_layer = n_layer
        self.latent_dim = latent_dim
    def forward(self,x):
        x = self.linear(x)
        return x.view(-1, self.n_layer, self.latent_dim)

    
class LevelsMapper(Module):

    def __init__(self):
        super(LevelsMapper, self).__init__()

        self.upscale = UpScaleEmbed()
        self.course_mapping = MapperBase()
        self.medium_mapping = MapperBase()
        self.fine_mapping = MapperBase()

    def forward(self, x):
        x = self.upscale(x)
        
        x_coarse = x[:, :5, :]
        x_medium = x[:, 5:12, :]
        x_fine = x[:, 12:, :]

        x_coarse = self.course_mapping(x_coarse)
        x_medium = self.medium_mapping(x_medium)
        x_fine = self.fine_mapping(x_fine)

        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)

        return out