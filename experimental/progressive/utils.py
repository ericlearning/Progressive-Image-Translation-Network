import os
import cv2
import glob
import torch
import imageio
import numpy as np
import pandas as pd
import seaborn as sn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.io import wavfile
from PIL import Image

def set_lr(optimizer, lrs):
	if(len(lrs) == 1):
		for param in optimizer.param_groups:
			param['lr'] = lrs[0]
	else:
		for i, param in enumerate(optimizer.param_groups):
			param['lr'] = lrs[i]

def get_lr(optimizer):
	optim_param_groups = optimizer.param_groups
	if(len(optim_param_groups) == 1):
		return optim_param_groups[0]['lr']
	else:
		lrs = []
		for param in optim_param_groups:
			lrs.append(param['lr'])
		return lrs

def histogram_sizes(img_dir, h_lim = None, w_lim = None):
	hs, ws = [], []
	for file in glob.iglob(os.path.join(img_dir, '**/*.*')):
		try:
			with Image.open(file) as im:
				h, w = im.size
				hs.append(h)
				ws.append(w)
		except:
			print('Not an Image file')

	if(h_lim is not None and w_lim is not None):
		hs = [h for h in hs if h<h_lim]
		ws = [w for w in ws if w<w_lim]

	plt.figure('Height')
	plt.hist(hs)

	plt.figure('Width')
	plt.hist(ws)

	plt.show()

	return hs, ws

def generate_noise(bs, nz, device):
	if(nz == None):
		return None
	noise = torch.randn(bs, nz, 1, 1, device = device)
	return noise

def get_display_samples(samples, num_samples_x, num_samples_y):
	sz = samples[0].shape[0]
	nc = samples[0].shape[2]
	display = np.zeros((sz*num_samples_y, sz*num_samples_x, nc))
	for i in range(num_samples_y):
		for j in range(num_samples_x):
			display[i*sz:(i+1)*sz, j*sz:(j+1)*sz, :] = cv2.cvtColor(samples[i*num_samples_x+j]*255.0, cv2.COLOR_BGR2RGB)
	return display.astype(np.uint8)

def save(filename, netD, netG, optD, optG):
	state = {
		'netD' : netD.state_dict(),
		'netG' : netG.state_dict(),
		'optD' : optD.state_dict(),
		'optG' : optG.state_dict()
	}
	torch.save(state, filename)

def load(filename, netD, netG, optD, optG):
	state = torch.load(filename)
	netD.load_state_dict(state['netD'])
	netG.load_state_dict(state['netG'])
	optD.load_state_dict(state['optD'])
	optG.load_state_dict(state['optG'])

def save_extra(filename, netD_1, netD_2, netD_3, netG, optD, optG):
	state = {
		'netD_1' : netD_1.state_dict(),
		'netD_2' : netD_2.state_dict(),
		'netD_3' : netD_3.state_dict(),
		'netG' : netG.state_dict(),
		'optD' : optD.state_dict(),
		'optG' : optG.state_dict()
	}
	torch.save(state, filename)

def load_extra(filename, netD_1, netD_2, netD_3, netG, optD, optG):
	state = torch.load(filename)
	netD_1.load_state_dict(state['netD_1'])
	netD_2.load_state_dict(state['netD_2'])
	netD_3.load_state_dict(state['netD_3'])
	netG.load_state_dict(state['netG'])
	optD.load_state_dict(state['optD'])
	optG.load_state_dict(state['optG'])

def get_sample_images_list(inputs):
	val_data, netG, p, device = inputs[0], inputs[1], inputs[2], inputs[3]
	with torch.no_grad():
		val_x = val_data[0].to(device)
		val_y = val_data[1].to(device)
		sample_input_images = val_x.detach().cpu().numpy() # l (C, H, W)
		sample_input_images_list = []
		sample_output_images = val_y.detach().cpu().numpy()# real ab
		sample_output_images_list = []
		sample_fake_images = netG(val_x, p).detach().cpu().numpy() # fake ab
		sample_fake_images_list = []
		sample_images_list = []

	for j in range(3):
		cur_img = (sample_fake_images[j] + 1) / 2.0
		sample_fake_images_list.append(cur_img.transpose(1, 2, 0))
	for j in range(3):
		cur_img = (sample_input_images[j] + 1) / 2.0
		sample_input_images_list.append(cur_img.transpose(1, 2, 0))
	for j in range(3):
		cur_img = (sample_output_images[j] + 1) / 2.0
		sample_output_images_list.append(cur_img.transpose(1, 2, 0))
	
	sample_images_list.extend(sample_input_images_list)
	sample_images_list.extend(sample_fake_images_list)
	sample_images_list.extend(sample_output_images_list)

	return sample_images_list

def get_sample_images_list_noise(inputs):
	val_data, netG, p, noise, device = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
	with torch.no_grad():
		val_x = val_data[0].to(device).repeat(3, 1, 1, 1)
		val_y = val_data[1].to(device).repeat(3, 1, 1, 1)
		noise = torch.cat([noise[0].unsqueeze(0)] * 3 + [noise[1].unsqueeze(0)] * 3 + [noise[2].unsqueeze(0)] * 3, 0)
		sample_input_images = val_x.detach().cpu().numpy() # l (C, H, W)
		sample_input_images_list = []
		sample_output_images = val_y.detach().cpu().numpy()# real ab
		sample_output_images_list = []
		sample_fake_images = netG(val_x, p, noise).detach().cpu().numpy() # fake ab
		sample_fake_images_list = []
		sample_images_list = []

	for j in range(9):
		cur_img = (sample_fake_images[j] + 1) / 2.0
		sample_fake_images_list.append(cur_img.transpose(1, 2, 0))
	for j in range(9):
		cur_img = (sample_input_images[j] + 1) / 2.0
		sample_input_images_list.append(cur_img.transpose(1, 2, 0))
	for j in range(9):
		cur_img = (sample_output_images[j] + 1) / 2.0
		sample_output_images_list.append(cur_img.transpose(1, 2, 0))
	
	sample_images_list.extend(sample_input_images_list)
	sample_images_list.extend(sample_fake_images_list)
	sample_images_list.extend(sample_output_images_list)

	return sample_images_list

def resize_input(x, y, fake_y):
	x1 = x																							# (sz, sz)
	x2 = F.adaptive_avg_pool2d(x, (x.shape[2] // 2, x.shape[3] // 2))								# (sz/2, sz/2)
	x3 = F.adaptive_avg_pool2d(x, (x.shape[3] // 4, x.shape[3] // 4))								# (sz/4, sz/4)

	y1 = y																							# (sz, sz)
	y2 = F.adaptive_avg_pool2d(y, (y.shape[2] // 2, y.shape[3] // 2))								# (sz/2, sz/2)
	y3 = F.adaptive_avg_pool2d(y, (y.shape[2] // 4, y.shape[3] // 4))								# (sz/4, sz/4)

	fake_y_1 = fake_y 																				# (sz, sz)
	fake_y_2 = F.adaptive_avg_pool2d(fake_y, (fake_y.shape[2] // 2, fake_y.shape[3] // 2))			# (sz/2, sz/2)
	fake_y_3 = F.adaptive_avg_pool2d(fake_y, (fake_y.shape[2] // 4, fake_y.shape[3] // 4))			# (sz/4, sz/4)

	return (x1, x2, x3), (y1, y2, y3), (fake_y_1, fake_y_2, fake_y_3)

def get_require_type(loss_type):
	if(loss_type == 'SGAN' or loss_type == 'LSGAN' or loss_type == 'HINGEGAN' or loss_type == 'WGAN'):
		require_type = 0
	elif(loss_type == 'RASGAN' or loss_type == 'RALSGAN' or loss_type == 'RAHINGEGAN'):
		require_type = 1
	elif(loss_type == 'QPGAN'):
		require_type = 2
	else:
		require_type = -1
	return require_type

def get_gan_loss(device, loss_type):
	loss_dict = {'SGAN':SGAN, 'LSGAN':LSGAN, 'HINGEGAN':HINGEGAN, 'WGAN':WGAN, 'RASGAN':RASGAN, 'RALSGAN':RALSGAN, 'RAHINGEGAN':RAHINGEGAN, 'QPGAN':QPGAN}
	require_type = get_require_type(loss_type)

	if(require_type == 0):
		loss = loss_dict[loss_type](device)
	elif(require_type == 1):
		loss = loss_dict[loss_type](device)
	elif(require_type == 2):
		loss = loss_dict[loss_type](device, 'L1')
	else:
		loss = None

	return loss
