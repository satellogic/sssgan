"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_satellite_model import Pix2PixSatelliteModel
from .class_dist_regularizer import SatelliteDistributionRegularizer
from torchvision import transforms
import torch
import util.util as util

class ClassDist:
    def __init__(self, opt):
        self.class_dist_model = SatelliteDistributionRegularizer(dp=DataParallelWithCallback, device_ids=opt.gpu_ids)
        
    def get_class_dist(self, data):
        inv_trans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                    transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                        std = [ 1., 1., 1. ]),
                                ])
        reg_transform = transforms.Normalize((0.2359, 0.1949 , 0.1598), (0.1176, 0.0961, 0.0864))
        
        real_image = torch.stack([reg_transform(inv_trans(img)) for img in data["image"]])
        with torch.no_grad():
            real_vec = self.class_dist_model.model(real_image).detach()
        data["class_dist"] = real_vec

class Pix2PixSatelliteTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = Pix2PixSatelliteModel(opt)
        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr
            if opt.continue_train:
                self.optimizer_G = util.load_optimizer(self.optimizer_G, 'G', opt.which_epoch, opt)
                self.optimizer_D = util.load_optimizer(self.optimizer_D, 'D', opt.which_epoch, opt)

        self.satellite_reg = opt.satellite_reg
        if self.satellite_reg:
            self.reg = SatelliteDistributionRegularizer(dp=DataParallelWithCallback, device_ids=opt.gpu_ids)
            self.lambda_reg = opt.lambda_reg

        self.inv_trans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

        self.reg_transform = transforms.Normalize((0.2359, 0.1949 , 0.1598), (0.1176, 0.0961, 0.0864))

        #Virtual batch
        self.batch_multiplayer = 12
        self.generator_counter = 0
        self.discriminator_counter = 0


    def run_generator_one_step(self, data):
        if self.generator_counter == 0:
            self.optimizer_G.zero_grad()
        if self.satellite_reg:
            real_image = self.reg_transform(self.inv_trans(data["image"]))
            #real_image = torch.stack([self.reg_transform(self.inv_trans(img)) for img in data["image"]])
            with torch.no_grad():
                real_vec = self.reg.model(real_image).detach()
            data["class_dist"] = real_vec
        g_losses, generated = self.pix2pix_model(data, mode='generator')
        values = list(g_losses.values())
        if self.satellite_reg:
            fake = self.reg_transform(self.inv_trans(generated))
            #fake = torch.stack([self.reg_transform(self.inv_trans(img)) for img in generated])
            self.reg.zero_grad()
            g_losses["SatReg"] = self.reg(data["class_dist"], fake)
            values.append(g_losses["SatReg"]*self.lambda_reg)
        g_loss = sum(values).mean()
        g_loss.backward()
        if self.generator_counter == 0:
            self.optimizer_G.step()
            self.generator_counter = self.batch_multiplayer
        self.generator_counter -= 1
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data):
        if self.discriminator_counter == 0:
            self.optimizer_D.zero_grad()
        d_losses = self.pix2pix_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        
        if self.discriminator_counter == 0:
            self.optimizer_D.step()
            self.discriminator_counter = self.batch_multiplayer
        self.discriminator_counter -= 1
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)
        util.save_optimizer(self.optimizer_G, "G", epoch, self.opt)
        util.save_optimizer(self.optimizer_D, "D", epoch, self.opt)


    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
