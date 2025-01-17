import torch
from torch import nn
import m_latent_mappers
from models.stylegan2.model import Generator
import math


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class StyleCLIPMapper(nn.Module):

    def __init__(self, checkpoint_path):
        super(StyleCLIPMapper, self).__init__()
        # Define architecture
        #self.checkpoint_path = checkpoint_path
        self.n_styles = int(math.log(1024, 2)) * 2 - 2
        self.mapper = self.set_mapper()
        self.decoder = Generator(1024, 512, 8)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights(checkpoint_path)
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_mapper(self):
        mapper = m_latent_mappers.LevelsMapper()
        return mapper

    def load_weights(self, path):
        print('Loading from checkpoint: {}'.format(path))
        ckpt = torch.load(path, map_location='cuda:0')
        self.mapper.load_state_dict(get_keys(ckpt, 'mapper'), strict=True)
        self.mapper = self.mapper.to(torch.device('cuda:0'))
        #self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
        ckpt = torch.load("./pretrained_models/stylegan2-ffhq-config-f.pt")
        self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
        self.decoder = self.decoder.to(torch.device('cuda:0'))
        self.__load_latent_avg(ckpt, repeat=self.n_styles)
            
    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            #self.latent_avg = ckpt['latent_avg'].to(self.device)
            self.latent_avg = ckpt['latent_avg'].to(torch.device('cuda:0'))
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

    def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):
        
        if input_code:
            codes = x
        else:
            codes = self.mapper(x)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = not input_code
        images, result_latent = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images


