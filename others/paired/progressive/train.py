import os
import torch
import torch.nn as nn
from torchvision import transforms
from dataset import Dataset
from architectures.architecture import UNet_G, PatchGan_D_70x70
from trainers.trainer import Trainer
from utils import save, load

train_dir_name = ['data/file/train/input', 'data/file/train/target']
val_dir_name = ['data/file/val/input', 'data/file/val/target']

lr_D, lr_G, bs = 0.0002, 0.0002, 8
sz, ic, oc, use_sigmoid = 256, 3, 3, False
norm_type = 'instancenorm'

train_data = Dataset(train_dir_name, basic_types = 'Pix2Pix', shuffle = True)
val_data = Dataset(val_dir_name, basic_types = 'Pix2Pix', shuffle = False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
netD = PatchGan_D_70x70(ic, oc, use_sigmoid, norm_type).to(device)
netG = UNet_G(ic, oc, sz, nz = 8, norm_type = norm_type).to(device)

trainer = Trainer('SGAN', netD, netG, device, train_data, val_data, lr_D = lr_D, lr_G = lr_G, rec_weight = 10, ds_weight = 8, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')

trainer.train([5, 5, 5, 5], [0.5, 0.5, 0.5], [16, 16, 16, 16])
save('saved/cur_state.state', netD, netG, trainer.optimizerD, trainer.optimizerG)