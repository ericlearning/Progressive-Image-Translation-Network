import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm

def get_norm(norm_type, size):
	if(norm_type == 'batchnorm'):
		return nn.BatchNorm2d(size)
	elif(norm_type == 'instancenorm'):
		return nn.InstanceNorm2d(size)

class Nothing(nn.Module):
	def __init__(self):
		super(Nothing, self).__init__()
		
	def forward(self, x):
		return x

class ConvBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad = None, pad_type = 'Zero', use_bn = True, use_sn = False, norm_type = 'batchnorm', activation_type = 'leakyrelu'):
		super(ConvBlock, self).__init__()
		self.use_bn = use_bn
		self.use_sn = use_sn
		self.norm_type = norm_type
		self.pad_type = pad_type

		if(pad == None):
			pad = ks // 2 // stride

		if(self.pad_type == 'Zero'):
			self.conv = nn.Conv1d(ni, no, ks, stride, pad, bias = False)
		elif(self.pad_type == 'Reflection'):
			self.conv = nn.Conv1d(ni, no, ks, stride, 0, bias = False)
			self.reflection = nn.ReflectionPad1d(pad)

		if(self.use_bn == True):
			if(self.norm_type == 'batchnorm'):
				self.bn = nn.BatchNorm1d(no)
			elif(self.norm_type == 'instancenorm'):
				self.bn = nn.InstanceNorm1d(no)

		if(self.use_sn == True):
			self.conv = SpectralNorm(self.conv)

		if(activation_type == 'relu'):
			self.act = nn.ReLU(inplace = True)
		elif(activation_type == 'leakyrelu'):
			self.act = nn.LeakyReLU(0.2, inplace = True)
		elif(activation_type == 'elu'):
			self.act = nn.ELU(inplace = True)
		elif(activation_type == 'selu'):
			self.act = nn.SELU(inplace = True)
		elif(activation_type == None):
			self.act = Nothing()

	def forward(self, x):
		out = x
		if(self.pad_type == 'Reflection'):
			out = self.reflection(out)
		out = self.conv(out)
		if(self.use_bn == True):
			out = self.bn(out)
		out = self.act(out)
		return out

class DeConvBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad = None, output_pad = 0, use_bn = True, use_sn = False, norm_type = 'batchnorm', activation_type = 'leakyrelu'):
		super(DeConvBlock, self).__init__()
		self.use_bn = use_bn
		self.use_sn = use_sn
		self.norm_type = norm_type

		if(pad is None):
			pad = ks // 2 // stride

		self.deconv = nn.ConvTranspose1d(ni, no, ks, stride, pad, output_padding = output_pad, bias = False)

		if(self.use_bn == True):
			if(self.norm_type == 'batchnorm'):
				self.bn = nn.BatchNorm1d(no)
			elif(self.norm_type == 'instancenorm'):
				self.bn = nn.InstanceNorm1d(no)

		if(self.use_sn == True):
			self.deconv = SpectralNorm(self.deconv)

		if(activation_type == 'relu'):
			self.act = nn.ReLU(inplace = True)
		elif(activation_type == 'leakyrelu'):
			self.act = nn.LeakyReLU(0.2, inplace = True)
		elif(activation_type == 'elu'):
			self.act = nn.ELU(inplace = True)
		elif(activation_type == 'selu'):
			self.act = nn.SELU(inplace = True)
		elif(activation_type == None):
			self.act = Nothing()

	def forward(self, x):
		out = self.deconv(x)
		if(self.use_bn == True):
			out = self.bn(out)
		out = self.act(out)
		return out
		
def receptive_calculator(input_size, ks, stride, pad):
	return int((input_size - ks + 2 * pad) / stride + 1)

def inverse_receptive_calculator(output_size, ks, stride, pad):
	return ((output_size - 1) * stride) + ks

class UNet_G(nn.Module):
	def __init__(self, ic = 1, oc = 1, use_bn = True, use_sn = False, norm_type = 'instancenorm'):
		super(UNet_G, self).__init__()
		self.ic = ic
		self.oc = oc
		self.use_bn = use_bn
		self.use_sn = use_sn

		self.leaky_relu = nn.LeakyReLU(0.2, inplace = True)
		self.relu = nn.ReLU(inplace = True)

		self.cur_dim = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]

		self.enc_convs = nn.ModuleList([])
		cur_block_ic = 1
		for i, dim in enumerate(self.cur_dim):
			if(i == 0):
				block = ConvBlock(self.ic, self.cur_dim[0], 32, 2, 15, use_bn = False, use_sn = self.use_sn, activation_type = None)
			elif(i == len(self.cur_dim) - 1):
				block = ConvBlock(self.cur_dim[i-1], self.cur_dim[i], 32, 2, 15, use_bn = False, use_sn = self.use_sn, activation_type = None)
			else:
				block = ConvBlock(self.cur_dim[i-1], self.cur_dim[i], 32, 2, 15, use_bn = True, norm_type = norm_type, use_sn = self.use_sn, activation_type = None)
			self.enc_convs.append(block)

		self.dec_convs = nn.ModuleList([])
		cur_block_ic = self.cur_dim[-1]
		for i, dim in enumerate(list(reversed(self.cur_dim))[1:] + [self.oc]):
			if(i == 0):
				de_block = DeConvBlock(cur_block_ic, dim, 32, 2, 15, use_bn = False, use_sn = self.use_sn, activation_type = None)
			elif(i == len(self.cur_dim) - 1):
				de_block = DeConvBlock(cur_block_ic*2, self.oc, 32, 2, 15, use_bn = False, use_sn = self.use_sn, activation_type = None)
			else:
				de_block = DeConvBlock(cur_block_ic*2, dim, 32, 2, 15, use_bn = self.use_bn, use_sn = self.use_sn, norm_type = norm_type, activation_type = None)
			cur_block_ic = dim
			self.dec_convs.append(de_block)

		self.tanh = nn.Tanh()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()
	
	def forward(self, x):
		ens = []
		out = x

		for i, cur_enc in enumerate(self.enc_convs):
			if(i == 0):
				out = cur_enc(out)
			else:
				out = cur_enc(self.leaky_relu(out))
			ens.append(out)

		for i, cur_dec in enumerate(self.dec_convs):
			cur_enc = ens[len(self.cur_dim) - 1 - i]
			if(i == 0):
				out = cur_dec(self.relu(cur_enc))
			else:
				out = cur_dec(self.relu(torch.cat([out, cur_enc], 1)))
				
		del ens
		out = self.tanh(out)
		return out