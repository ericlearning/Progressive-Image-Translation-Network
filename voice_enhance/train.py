import os
import torch
import torch.nn as nn
from torchvision import transforms
from dataset import Dataset
from architectures.baseline.unet import UNet_G
from architectures.virtual.virtual_unet import Virtual_UNet_G
from architectures.baseline.discriminator import PatchGan_D_70x70
from trainers.trainer import Trainer
from utils.utils import save, load

train_dir_name = ['data/file/train/input', 'data/file/train/target']
val_dir_name = ['data/file/val/input', 'data/file/val/target']

lr_D, lr_G, bs = 0.0002, 0.0002, 8
ic, oc, use_sigmoid = 1, 1, False
norm_type = 'instancenorm'

train_data = Dataset(train_dir_name, basic_types = 'CycleGan', shuffle = True, single_channel = False)
val_data = Dataset(val_dir_name, basic_types = 'Pix2Pix', shuffle = False, single_channel = False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

netD = PatchGan_D_70x70(ic, oc, use_sigmoid, norm_type, use_sn = False).to(device)
netG = UNet_G(ic, oc, use_bn = True, use_sn = False, norm_type = norm_type).to(device)

trn_dl = train_data.get_loader(None, bs)
val_dl = list(val_data.get_loader(None, 3))[0]

trainer = Trainer('SGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, rec_weight = 10, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_images/'):
trainer.train(5)
save('saved/cur_state.state', netD, netG, trainer.optimizerD, trainer.optimizerG)