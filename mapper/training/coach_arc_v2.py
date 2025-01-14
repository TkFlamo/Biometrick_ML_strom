import os

import clip
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# import criteria.clip_loss as clip_loss
from criteria.lpips.lpips import LPIPS
from criteria import id_loss, w_norm
import torch.nn.functional as F
from mapper.datasets.latents_dataset import LatentsDataset, StyleSpaceLatentsDataset, EmbedImagePairs
from mapper.styleclip_mapper import StyleCLIPMapper
from mapper.training.ranger import Ranger
from mapper.training import train_utils
from mapper.training.train_utils import convert_s_tensor_to_list
from torchvision import transforms


class Coach:
    def __init__(self, opts):
        self.opts = opts

        self.global_step = 0

        self.device = 'cuda:0'
        self.opts.device = self.device

        # Initialize network
        self.net = StyleCLIPMapper(self.opts).to(self.device)
        self.avg_latent = self.net.latent_avg
        #self.avg_latent = torch.load('./optimization/ref_latent.pt')[0].cuda()
        #self.avg_latent = torch.load('./optimization/ref_latent_short_hair.pt')[0].cuda()
        with torch.no_grad():
            self.ref_img, _, _ = self.net.decoder([self.avg_latent.unsqueeze(0)], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
            self.ref_img = self.net.face_pool(self.ref_img)
        #print(f'### {self.ref_img.shape}')
        # Initialize loss
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss(self.opts).to(self.device).eval()
        if self.opts.l2_lambda > 0:
            self.l2_loss = nn.MSELoss()


        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)

        #self.text_inputs = torch.cat([clip.tokenize(self.opts.description)]).cuda()

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

    def train(self):
        self.net.train()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                
                if self.opts.work_in_stylespace:
                    w = convert_s_tensor_to_list(batch)
                    w = [c.to(self.device) for c in w]
                else:
                    x, embed = batch
                    x, embed = x.to(self.device), embed.to(self.device)
                with torch.no_grad():
                    #embed = self.id_loss.extract_feats(x)
                    #print(f'embed shape {embed.shape}')
                    avg_latent = self.avg_latent.repeat(embed.shape[0], 1, 1)
                    ref_img = self.ref_img.repeat(embed.shape[0], 1, 1, 1)

                if self.opts.work_in_stylespace:
                    delta = self.net.mapper(w)
                    w_hat = [c + 0.1 * delta_c for (c, delta_c) in zip(w, delta)]
                    x_hat, _, w_hat = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1, input_is_stylespace=True)
                else:
                    w = self.net.mapper(embed)
                    
                    x_hat, _, _ = self.net.decoder([w], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
                    x_hat = self.net.face_pool(x_hat)

                loss, loss_dict = self.calc_loss(x, x_hat,ref_img,w)
                loss.backward()
                self.optimizer.step()

                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 1000 == 0):
                    self.parse_and_log_images(x, x_hat, title='images_train')
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            if batch_idx > 200:
                break

            if self.opts.work_in_stylespace:
                w = convert_s_tensor_to_list(batch)
                w = [c.to(self.device) for c in w]
            else:
                x, embed = batch
                #print('x validate shape: ', x.shape)
                x, embed = x.to(self.device), embed.to(self.device)
            
            with torch.no_grad():
                #embed = self.id_loss.extract_feats(x)
                avg_latent = self.avg_latent.repeat(embed.shape[0], 1, 1)
                ref_img = self.ref_img.repeat(embed.shape[0], 1, 1, 1)
                #x, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1, input_is_stylespace=self.opts.work_in_stylespace)
                if self.opts.work_in_stylespace:
                    delta = self.net.mapper(w)
                    w_hat = [c + 0.1 * delta_c for (c, delta_c) in zip(w, delta)]
                    x_hat, _, w_hat = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1, input_is_stylespace=True)
                else:
                    w = self.net.mapper(embed)
                    x_hat, _, _ = self.net.decoder([w], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
                    x_hat = self.net.face_pool(x_hat)
                
                loss, cur_loss_dict = self.calc_loss(x, x_hat,ref_img,w)
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            self.parse_and_log_images(x, x_hat, title='images_val', index=batch_idx)

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        params = list(self.net.mapper.parameters())
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        full_dataset = EmbedImagePairs(img_dir=self.opts.img_dir,
                                       embed_dir=self.opts.embed_dir,
                                       opts=self.opts)
        print('full dataset size: ', len(full_dataset))
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [int(len(full_dataset)*0.9), int(len(full_dataset)*0.1)])
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        
        return train_dataset, test_dataset

    def calc_loss(self, x, x_hat,ref_img,w):
        loss_dict = {}
        loss = 0.0
        if self.opts.id_lambda > 0:
            loss_id = self.id_loss(x_hat, x)[0]
            loss_dict['loss_id'] = float(loss_id)
            #loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = self.l2_loss(x_hat, x)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda #changing this one for Objective #2
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(x_hat, x)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda #changing this one for Objective #2
        if self.opts.l2w_lambda > 0:
            #loss_l2w = ((self.avg_latent - w) ** 2).sum()/w.shape[0] * 0.01
            loss_l2w = torch.sum((self.avg_latent - w).norm(2, dim=(1, 2))) / w.shape[0]
            loss_dict['l2w_lambda'] = float(loss_l2w)
            loss += loss_l2w * self.opts.l2w_lambda
        if self.opts.w_norm_lambda > 0:
            loss_w_norm = self.w_norm_loss(w, self.net.latent_avg)
            loss_dict['loss_w_norm'] = float(loss_w_norm)
            loss += loss_w_norm * self.opts.w_norm_lambda
        #print(x_hat.shape)
        #loss_inter = 1 - self.id_loss(x_hat[0].unsqueeze(0), x_hat[1].unsqueeze(0))
        #loss_dict['loss_inter'] = float(loss_inter)
        #loss += loss_inter * 0.1
        loss_dict['loss'] = float(loss)
        return loss, loss_dict

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            #pass
            print(f"step: {self.global_step} \t metric: {prefix}/{key} \t value: {value}")
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, x, x_hat, title, index=None):
        if index is None:
            path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}.jpg')
        else:
            path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}_{str(index).zfill(5)}.jpg')
        path_1 = path[:-4] + "_gen.jpg"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        p = transforms.Compose([transforms.Resize((256,256))])
        #log_img = torch.cat([p(x).detach().cpu(), 
        #                     p(x_hat).detach().cpu()])
        torchvision.utils.save_image(p(p(x).detach().cpu()), path,
                                     normalize=True, scale_each=True, 
                                     #range=(-1, 1), 
                                     nrow=self.opts.batch_size)
        torchvision.utils.save_image(p(p(x_hat).detach().cpu()), path_1,
                                     normalize=True, scale_each=True, 
                                     #range=(-1, 1), 
                                     nrow=self.opts.batch_size)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts)
        }
        save_dict['latent_avg'] = self.net.latent_avg
#         if self.opts.start_from_latent_avg:
#             save_dict['latent_avg'] = self.net.latent_avg
        return save_dict