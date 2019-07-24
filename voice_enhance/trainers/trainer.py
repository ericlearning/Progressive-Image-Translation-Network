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
from utils.utils import *
from losses.losses import *

class Trainer():
	def __init__(self, loss_type, netD, netG, device, train_dl, val_dl, lr_D = 0.0002, lr_G = 0.0002, rec_weight = 10, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 50, image_interval = 50, save_img_dir = 'saved_images/'):
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

		self.loss_type = loss_type
		self.require_type = get_require_type(self.loss_type)
		self.loss = get_gan_loss(self.device, self.loss_type)

		self.optimizerD = optim.Adam(self.netD.parameters(), lr = self.lr_D, betas = (0.5, 0.999))
		self.optimizerG = optim.Adam(self.netG.parameters(), lr = self.lr_G, betas = (0.5, 0.999))

		self.loss_interval = loss_interval
		self.image_interval = image_interval

		self.save_cnt = 0
		self.save_img_dir = save_img_dir
		if(not os.path.exists(self.save_img_dir)):
			os.makedirs(self.save_img_dir)

	def gradient_penalty(self, x, real_image, fake_image):
		raise NotImplementedError

	def train(self, num_epoch):
		l1 = nn.L1Loss()

		for epoch in range(num_epoch):
			if(self.resample):
				train_dl_iter = iter(self.train_dl)
				
			for i, (x, y) in enumerate(tqdm(self.train_dl)):
				x = x.to(self.device)
				y = y.to(self.device)
				bs = x.size(0)
				fake_y = self.netG(x)

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
					fake_y = self.netG(x)

				if(self.require_type == 0):
					c_xr = None
					c_xf = self.netD(x, fake_y)		# (bs, 1, 1, 1)
					c_xf = c_xf.view(-1)						# (bs)	
					errG_1 = self.loss.g_loss(c_xf)
				if(self.require_type == 1 or self.require_type == 2):
					c_xr = self.netD(x, y)				# (bs, 1, 1, 1)
					c_xr = c_xr.view(-1)						# (bs)
					c_xf = self.netD(x, fake_y)		# (bs, 1, 1, 1)
					c_xf = c_xf.view(-1)						# (bs)		
					errG_1 = self.loss.g_loss(c_xr, c_xf)
				
				if(self.rec_weight == 0):
					rec_loss = 0
				else:
					rec_loss = l1(fake_y, y)

				errG = errG_1 + rec_loss * self.rec_weight
				errG.backward()
				#update G using the gradients calculated previously
				self.optimizerG.step()

				if(i % self.loss_interval == 0):
					print('[%d/%d] [%d/%d] errD : %.4f, errG : %.4f, L1 : %.4f'
						  %(epoch+1, num_epoch, i+1, self.train_iteration_per_epoch, errD, errG, rec_loss))

				if(i % self.image_interval == 0):
					sample_images_list = get_sample_images_list((self.val_dl, self.netG, self.device))
					plot_image = get_display_samples(sample_images_list, 3, 3)
						
					cur_file_name = os.path.join(self.save_img_dir, str(self.save_cnt)+' : '+str(epoch)+'-'+str(i)+'.jpg')
					self.save_cnt += 1
					cv2.imwrite(cur_file_name, plot_image)

			