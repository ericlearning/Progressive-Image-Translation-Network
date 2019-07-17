import os
import torch
import torch.nn as nn
from torchvision import transforms
from dataset import Dataset
from architectures.baseline.unet import UNet_G
from architectures.baseline.resnet import ResNet_G
from architectures.baseline.spade import SPADE_G
from architectures.baseline.discriminator import PatchGan_D_70x70_One_Input
from trainers.trainer import Trainer
from utils.utils import save, load

train_dir_name = ['data/file/train/input', 'data/file/train/target']
val_dir_name = ['data/file/val/input', 'data/file/val/target']

lr_D, lr_G, bs = 0.0002, 0.0002, 8
sz, ic, oc, use_sigmoid = 256, 3, 3, False
norm_type = 'instancenorm'

train_data = Dataset(train_dir_name, basic_types = 'CycleGan', shuffle = True, single_channel = False)
val_data = Dataset(val_dir_name, basic_types = 'Pix2Pix', shuffle = False, single_channel = False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

netD_A = PatchGan_D_70x70_One_Input(ic, use_sigmoid, norm_type, use_sn = False).to(device)
netD_B = PatchGan_D_70x70_One_Input(oc, use_sigmoid, norm_type, use_sn = False).to(device)
netG_A2B = ResNet_G(ic, oc, sz, nz = 8, norm_type = norm_type).to(device)
netG_B2A = ResNet_G(oc, ic, sz, nz = 8, norm_type = norm_type).to(device)

# netG_A2B = UNet_G(ic, oc, sz, nz = 8, norm_type = norm_type).to(device)
# netG_B2A = UNet_G(oc, ic, sz, nz = 8, norm_type = norm_type).to(device)
# netG_A2B = SPADE_G(ic, oc, sz, nz = 8).to(device)
# netG_B2A = SPADE_G(oc, ic, sz, nz = 8).to(device)

trn_dl = train_data.get_loader(256, bs)
val_dl = list(val_data.get_loader(256, 3))[0]

trainer = Trainer('SGAN', netD_A, netD_B, netG_A2B, netG_B2A, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, cycle_weight = 10, identity_weight = 5.0, ds_weight = 8, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')

trainer.train(5)
save('saved/cur_state.state', netD_A, netD_B, netG_A2B, netG_B2A, trainer.optimizerD_A, trainer.optimizerD_B, trainer.optimizerG)