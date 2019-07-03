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
from utils import set_lr, get_lr, generate_noise, save, get_sample_images_list, get_sample_images_list_noise, get_display_samples, resize_input
from utils import get_gan_loss, get_require_type
from losses.losses import *

class Trainer():
	def __init__(self, loss_type, netD, netG, device, train_dl, val_dl, lr_D = 0.0002, lr_G = 0.0002, rec_weight = 10, ds_weight = 8, use_rec_feature = False, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 50, image_interval = 50, save_img_dir = 'saved_images/'):
		self.netD = netD
		self.netG = netG
		self.train_dl = train_dl
		self.val_dl = val_dl
		self.lr_D = lr_D
		self.lr_G = lr_G
		self.train_iteration_per_epoch = len(self.train_dl)
		self.device = device
		self.resample = resample
		self.weight_clip = weight_clip
		self.use_gradient_penalty = use_gradient_penalty
		self.rec_weight = rec_weight
		self.use_rec_feature = use_rec_feature
		self.ds_weight = ds_weight

		self.nz = self.netG.nz
		self.fixed_noise = generate_noise(3, self.nz, self.device)

		self.loss_type = loss_type
		self.require_type = get_require_type(self.loss_type)
		self.loss = get_gan_loss(self.loss_type)
		self.ds_loss = DSGAN_Loss(self.device, self.nz)

		self.optimizerD = optim.Adam(self.netD.parameters(), lr = self.lr_D, betas = (0, 0.9))
		self.optimizerG = optim.Adam(self.netG.parameters(), lr = self.lr_G, betas = (0, 0.9))

		self.loss_interval = loss_interval
		self.image_interval = image_interval

		self.save_cnt = 0
		self.save_img_dir = save_img_dir
		if(not os.path.exists(self.save_img_dir)):
			os.makedirs(self.save_img_dir)

	def gradient_penalty(self, x, real_image, fake_image):
		bs = real_image.size(0)
		alpha = torch.FloatTensor(bs, 1, 1, 1).uniform_(0, 1).expand(real_image.size()).to(self.device)
		interpolation = alpha * real_image + (1 - alpha) * fake_image

		c_xi = self.netD(x, interpolation)
		gradients = autograd.grad(c_xi, interpolation, torch.ones(c_xi.size()).to(self.device),
								  create_graph = True, retain_graph = True, only_inputs = True)[0]
		gradients = gradients.view(bs, -1)
		penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
		return penalty

	def train(self, num_epoch):
		l1 = nn.L1Loss()

		for epoch in range(num_epoch):
			if(self.resample):
				train_dl_iter = iter(self.train_dl)
				
			for i, (x, y) in enumerate(tqdm(self.train_dl)):
				x = x.to(self.device)
				y = y.to(self.device)
				bs = x.size(0)
				noise = generate_noise(bs, self.nz, self.device)
				fake_y = self.netG(x, noise)

				self.netD.zero_grad()

				c_xr = self.netD(x, y)
				c_xr = c_xr.view(-1)
				c_xf = self.netD(x, fake_y.detach())
				c_xf = c_xf.view(-1)

				if(self.require_type == 0 or self.require_type == 1):
					errD = self.loss.d_loss(c_xr, c_xf)
				elif(self.require_type == 2):
					errD = self.loss.d_loss(c_xr, c_xf, y, fake_y)
				
				if(self.use_gradient_penalty != False):
					errD += self.use_gradient_penalty * self.gradient_penalty(x, y, fake_y)

				errD.backward()
				self.optimizerD.step()

				if(self.weight_clip != None):
					for param in self.netD.parameters():
						param.data.clamp_(-self.weight_clip, self.weight_clip)


				self.netG.zero_grad()
				if(self.resample):
					x, y = next(train_dl_iter)
					x = x.to(self.device)
					y = y.to(self.device)
					bs = x.size(0)
					noise = generate_noise(bs, self.nz, self.device)
					fake_y = self.netG(x, noise)

				if(self.require_type == 0):
					c_xr = None
					c_xf, f1 = self.netD(x, fake_y, True)		# (bs, 1, 1, 1)
					c_xf = c_xf.view(-1)						# (bs)	
					errG_1 = self.loss.g_loss(c_xf)
				if(self.require_type == 1 or self.require_type == 2):
					c_xr, f2 = self.netD(x, y, True)				# (bs, 1, 1, 1)
					c_xr = c_xr.view(-1)						# (bs)
					c_xf, f1 = self.netD(x, fake_y, True)		# (bs, 1, 1, 1)
					c_xf = c_xf.view(-1)						# (bs)		
					errG_1 = self.loss.g_loss(c_xr, c_xf)

				if(self.ds_weight == 0):
					ds_loss = 0
				else:
					noise1 = generate_noise(bs, self.nz, self.device)
					noise2 = generate_noise(bs, self.nz, self.device)
					fake_y1 = self.netG(x, noise1)
					fake_y2 = self.netG(x, noise2)
					ds_loss = self.ds_loss(fake_y1, fake_y2, noise1, noise2)
				
				if(self.rec_weight == 0):
					rec_loss = 0
				else:
					if(self.use_rec_feature):
						rec_loss = 0
						if(c_xr == None):
							c_xr, f2 = self.netD(x, y, True)				# (bs, 1, 1, 1)
							c_xr = c_xr.view(-1)						# (bs)
							for f1_, f2_ in zip(f1, f2):
								rec_loss += (f1_ - f2_).abs().mean()
							rec_loss /= len(f1)

					else:
						rec_loss = l1(fake_y, y)

				errG = errG_1 + rec_loss * self.rec_weight + ds_loss * self.ds_weight
				errG.backward()
				#update G using the gradients calculated previously
				self.optimizerG.step()

				if(i % self.loss_interval == 0):
					print('[%d/%d] [%d/%d] errD : %.4f, errG : %.4f'
						  %(epoch+1, num_epoch, i+1, self.train_iteration_per_epoch, errD, errG))

				if(i % self.image_interval == 0):
					if(self.nz == None):
						sample_images_list = get_sample_images_list((self.val_dl, self.netG, self.device))
						plot_image = get_display_samples(sample_images_list, 3, 3)
					else:
						sample_images_list = get_sample_images_list_noise((self.val_dl, self.netG, self.fixed_noise, self.device))
						plot_image = get_display_samples(sample_images_list, 9, 3)
						
					cur_file_name = os.path.join(self.save_img_dir, str(self.save_cnt)+' : '+str(epoch)+'-'+str(i)+'.jpg')
					self.save_cnt += 1
					cv2.imwrite(cur_file_name, plot_image)

			