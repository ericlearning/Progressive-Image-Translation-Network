import os, cv2
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from utils.utils import set_lr, get_lr, generate_noise, save, get_sample_images_list, get_sample_images_list_noise, get_display_samples, resize_input
from utils.utils import get_gan_loss, get_require_type
from losses.losses import *
from itertools import chain

class Trainer():
	def __init__(self, loss_type, netD_A, netD_B, netD_A_2, netD_B_2, netG_A2B, netG_B2A, device, train_dl, val_dl, lr_D = 0.0002, lr_G = 0.0002, cycle_weight = 10, identity_weight = 5.0, ds_weight = 8, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 50, image_interval = 50, save_img_dir = 'saved_images/'):
		self.netD_A = netD_A
		self.netD_B = netD_B
		self.netD_A_2 = netD_A_2
		self.netD_B_2 = netD_B_2

		self.netG_A2B = netG_A2B
		self.netG_B2A = netG_B2A
		self.train_dl = train_dl
		self.val_dl = val_dl
		self.lr_D = lr_D
		self.lr_G = lr_G
		self.train_iteration_per_epoch = len(self.train_dl)
		self.device = device
		self.resample = resample
		self.weight_clip = weight_clip
		self.use_gradient_penalty = use_gradient_penalty
		self.cycle_weight = cycle_weight
		self.identity_weight = identity_weight
		self.ds_weight = ds_weight

		self.nz = self.netG_A2B.nz
		self.fixed_noise = generate_noise(3, self.nz, self.device)

		self.loss_type = loss_type
		self.require_type = get_require_type(self.loss_type)
		self.loss = get_gan_loss(self.device, self.loss_type)
		self.ds_loss = DSGAN_Loss(self.device, self.nz)

		self.optimizerD_A = optim.Adam(chain(self.netD_A.parameters(), self.netD_A_2.parameters()), lr = self.lr_D, betas = (0.5, 0.999))
		self.optimizerD_B = optim.Adam(chain(self.netD_B.parameters(), self.netD_B_2.parameters()), lr = self.lr_D, betas = (0.5, 0.999))
		self.optimizerG = optim.Adam(chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()), lr = self.lr_G, betas = (0.5, 0.999))

		self.loss_interval = loss_interval
		self.image_interval = image_interval

		self.save_cnt = 0
		self.save_img_dir = save_img_dir
		if(not os.path.exists(self.save_img_dir)):
			os.makedirs(self.save_img_dir)

	def train(self, num_epoch):
		l1 = nn.L1Loss()
		for epoch in range(num_epoch):
			if(self.resample):
				train_dl_iter = iter(self.train_dl)
			for i, (a, b) in enumerate(tqdm(self.train_dl)):
				a = a.to(self.device)
				b = b.to(self.device)
				bs = a.size(0)
				noise = generate_noise(bs, self.nz, self.device)
				fake_a = self.netG_B2A(b, noise)
				fake_b = self.netG_A2B(a, noise)



				self.optimizerD_A.zero_grad()
				c_xr_1 = self.netD_A(a)
				c_xr_1 = c_xr_1.view(-1)
				c_xf_1 = self.netD_A(fake_a.detach())
				c_xf_1 = c_xf_1.view(-1)
				errD_A_1 = self.loss.d_loss(c_xr_1, c_xf_1)
				c_xr_2 = self.netD_A_2(a)
				c_xr_2 = c_xr_2.view(-1)
				c_xf_2 = self.netD_A_2(fake_a.detach())
				c_xf_2 = c_xf_2.view(-1)
				errD_A_2 = self.loss.d_loss(c_xr_2, c_xf_2)
				errD_A = (errD_A_1 + errD_A_2) / 2.0
				errD_A.backward()
				self.optimizerD_A.step()



				self.optimizerD_B.zero_grad()
				c_xr_1 = self.netD_B(b)
				c_xr_1 = c_xr_1.view(-1)
				c_xf_1 = self.netD_B(fake_b.detach())
				c_xf_1 = c_xf_1.view(-1)
				errD_B_1 = self.loss.d_loss(c_xr_1, c_xf_1)
				c_xr_2 = self.netD_B_2(b)
				c_xr_2 = c_xr_2.view(-1)
				c_xf_2 = self.netD_B_2(fake_b.detach())
				c_xf_2 = c_xf_2.view(-1)
				errD_B_2 = self.loss.d_loss(c_xr_2, c_xf_2)
				errD_B = (errD_B_1 + errD_B_2) / 2.0
				errD_B.backward()
				self.optimizerD_B.step()



				if(self.weight_clip != None):
					for param in self.netD_A.parameters():
						param.data.clamp_(-self.weight_clip, self.weight_clip)
					for param in self.netD_A_2.parameters():
						param.data.clamp_(-self.weight_clip, self.weight_clip)

				if(self.weight_clip != None):
					for param in self.netD_B.parameters():
						param.data.clamp_(-self.weight_clip, self.weight_clip)
					for param in self.netD_B_2.parameters():
						param.data.clamp_(-self.weight_clip, self.weight_clip)




				self.optimizerG.zero_grad()
				if(self.resample):
					a, b = next(train_dl_iter)
					a = a.to(self.device)
					b = b.to(self.device)
					bs = a.size(0)
					noise = generate_noise(bs, self.nz, self.device)
					fake_a = self.netG_B2A(b, noise)
					fake_b = self.netG_A2B(a, noise)




				cycle_a = self.netG_B2A(fake_b, noise)
				cycle_b = self.netG_A2B(fake_a, noise)
				identity_a = self.netG_B2A(a, noise)
				identity_b = self.netG_A2B(b, noise)




				if(self.require_type == 0):
					c_xr_a_1 = None
					c_xr_a_2 = None
					c_xr_b_1 = None
					c_xr_b_2 = None

					c_xf_a_1 = self.netD_A(fake_a)
					c_xf_a_1 = c_xf_a_1.view(-1)
					c_xf_a_2 = self.netD_A_2(fake_a)
					c_xf_a_2 = c_xf_a_2.view(-1)
					c_xf_b_1 = self.netD_B(fake_b)
					c_xf_b_1 = c_xf_b_1.view(-1)
					c_xf_b_2 = self.netD_B_2(fake_b)
					c_xf_b_2 = c_xf_b_2.view(-1)

					errG_a_1 = self.loss.g_loss(c_xf_a_1)
					errG_a_2 = self.loss.g_loss(c_xf_a_2)
					errG_b_1 = self.loss.g_loss(c_xf_b_1)
					errG_b_2 = self.loss.g_loss(c_xf_b_2)

					errG_a = (errG_a_1 + errG_a_2) / 2.0
					errG_b = (errG_b_1 + errG_b_2) / 2.0


				if(self.require_type == 1 or self.require_type == 2):
					c_xr_a_1 = self.netD_A(a)
					c_xr_a_1 = c_xr_a_1.view(-1)
					c_xr_a_2 = self.netD_A_2(a)
					c_xr_a_2 = c_xr_a_2.view(-1)
					c_xr_b_1 = self.netD_B(b)
					c_xr_b_1 = c_xr_b_1.view(-1)
					c_xr_b_2 = self.netD_B_2(b)
					c_xr_b_2 = c_xr_b_2.view(-1)

					c_xf_a_1 = self.netD_A(fake_a)
					c_xf_a_1 = c_xf_a_1.view(-1)
					c_xf_a_2 = self.netD_A_2(fake_a)
					c_xf_a_2 = c_xf_a_2.view(-1)
					c_xf_b_1 = self.netD_B(fake_b)
					c_xf_b_1 = c_xf_b_1.view(-1)
					c_xf_b_2 = self.netD_B_2(fake_b)
					c_xf_b_2 = c_xf_b_2.view(-1)

					errG_a_1 = self.loss.g_loss(c_xr_a_1, c_xf_a_1)
					errG_a_2 = self.loss.g_loss(c_xr_a_2, c_xf_a_2)
					errG_b_1 = self.loss.g_loss(c_xr_b_1, c_xf_b_1)
					errG_b_2 = self.loss.g_loss(c_xr_b_2, c_xf_b_2)

					errG_a = (errG_a_1 + errG_a_2) / 2.0
					errG_b = (errG_b_1 + errG_b_2) / 2.0


				cycle_a_loss = l1(cycle_a, a)
				cycle_b_loss = l1(cycle_b, b)
				identity_a_loss = l1(identity_a, a)
				identity_b_loss = l1(identity_b, b)

				if(self.ds_weight == 0):
					ds_loss = 0
				else:
					noise1 = generate_noise(bs, self.nz, self.device)
					noise2 = generate_noise(bs, self.nz, self.device)
					fake_a1 = self.netG_B2A(b, noise1)
					fake_a2 = self.netG_B2A(b, noise2)
					fake_b1 = self.netG_A2B(a, noise1)
					fake_b2 = self.netG_A2B(a, noise2)
					ds_loss1 = self.ds_loss.get_loss(fake_a1, fake_a2, noise1, noise2)
					ds_loss2 = self.ds_loss.get_loss(fake_b1, fake_b2, noise1, noise2)
					ds_loss = (ds_loss1 + ds_loss2) / 2.0

				errG = errG_a + errG_b + (cycle_a_loss + cycle_b_loss) * self.cycle_weight + (identity_a_loss + identity_b_loss) * self.identity_weight
				errG = errG + ds_loss * self.ds_weight
				errG.backward()
				#update G using the gradients calculated previously
				self.optimizerG.step()

				if(i % self.loss_interval == 0):
					print('[%d/%d] [%d/%d] errD_A : %.4f, errD_B : %.4f, errG : %.4f'
						  %(epoch+1, num_epoch, i+1, self.train_iteration_per_epoch, errD_A, errD_B, errG))

				if(i % self.image_interval == 0):
					if(self.nz == None):
						sample_images_list = get_sample_images_list((self.val_dl, self.netG_A2B, self.netG_B2A, self.device))
						plot_image = get_display_samples(sample_images_list, 6, 3)
					else:
						sample_images_list = get_sample_images_list_noise((self.val_dl, self.netG_A2B, self.netG_B2A, self.fixed_noise, self.device))
						plot_image = get_display_samples(sample_images_list, 9, 6)
						
					cur_file_name = os.path.join(self.save_img_dir, str(self.save_cnt)+' : '+str(epoch)+'-'+str(i)+'.jpg')
					self.save_cnt += 1
					cv2.imwrite(cur_file_name, plot_image)

			