).import os
import cv2
import glob
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from scipy.io import wavfile
from PIL import Image
from losses.losses import *
from utils.griffin_lim import *

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
			if(nc == 1):
				display[i*sz:(i+1)*sz, j*sz:(j+1)*sz, :] = samples[i*num_samples_x+j]
			else:
				display[i*sz:(i+1)*sz, j*sz:(j+1)*sz, :] = cv2.cvtColor(samples[i*num_samples_x+j], cv2.COLOR_BGR2RGB)
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

def get_sample_images_list(inputs):
	# hard coded variables
	n_fft, win_length, hop_length, sample_rate, n_mels, power, shrink_size, threshold = 2048, 1000, 250, 22050, 256, 1, 1, 5

	val_data, netG, device = inputs[0], inputs[1], inputs[2]
	with torch.no_grad():
		val_x = val_data[0].to(device)
		val_y = val_data[1].to(device)
		sample_input_images = val_x.detach().cpu().numpy() # (bs, 1, sz)
		sample_input_images_list = []
		sample_output_images = val_y.detach().cpu().numpy()# (bs, 1, sz)
		sample_output_images_list = []
		sample_fake_images = netG(val_x).detach().cpu().numpy() # (bs, 1, sz)
		sample_fake_images_list = []
		sample_images_list = []

	for j in range(3):
		cur_img = mel_to_spectrogram(get_mel(get_stft(sample_fake_images[j].reshape(-1), \
			n_fft, win_length, hop_length), sample_rate, n_fft, n_mels, power, shrink_size), threshold, None)
		sample_fake_images_list.append(cv2.resize(cur_img, (256, 256)).reshape(256, 256, 1))
	for j in range(3):
		cur_img = mel_to_spectrogram(get_mel(get_stft(sample_input_images[j].reshape(-1), \
			n_fft, win_length, hop_length), sample_rate, n_fft, n_mels, power, shrink_size), threshold, None)
		sample_input_images_list.append(cv2.resize(cur_img, (256, 256)).reshape(256, 256, 1))
	for j in range(3):
		cur_img = mel_to_spectrogram(get_mel(get_stft(sample_output_images[j].reshape(-1), \
			n_fft, win_length, hop_length), sample_rate, n_fft, n_mels, power, shrink_size), threshold, None)
		sample_output_images_list.append(cv2.resize(cur_img, (256, 256)).reshape(256, 256, 1))
	
	sample_images_list.extend(sample_input_images_list)
	sample_images_list.extend(sample_fake_images_list)
	sample_images_list.extend(sample_output_images_list)

	return sample_images_list

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

