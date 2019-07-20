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
	def __init__(self, ni, no, ks, stride, pad = None, pad_type = 'Zero', use_bn = True, use_sn = False, use_pixelshuffle = False, norm_type = 'batchnorm', activation_type = 'leakyrelu'):
		super(ConvBlock, self).__init__()
		self.use_bn = use_bn
		self.use_sn = use_sn
		self.use_pixelshuffle = use_pixelshuffle
		self.norm_type = norm_type
		self.pad_type = pad_type

		if(pad == None):
			pad = ks // 2 // stride

		if(use_pixelshuffle):
			if(self.pad_type == 'Zero'):
				self.conv = nn.Conv2d(ni, no * 4, ks, stride, pad, bias = False)
			elif(self.pad_type == 'Reflection'):
				self.conv = nn.Conv2d(ni, no * 4, ks, stride, 0, bias = False)
				self.reflection = nn.ReflectionPad2d(pad)
			self.pixelshuffle = nn.PixelShuffle(2)
		else:
			if(self.pad_type == 'Zero'):
				self.conv = nn.Conv2d(ni, no, ks, stride, pad, bias = False)
			elif(self.pad_type == 'Reflection'):
				self.conv = nn.Conv2d(ni, no, ks, stride, 0, bias = False)
				self.reflection = nn.ReflectionPad2d(pad)

		if(self.use_bn == True):
			if(self.norm_type == 'batchnorm'):
				self.bn = nn.BatchNorm2d(no)
			elif(self.norm_type == 'instancenorm'):
				self.bn = nn.InstanceNorm2d(no)

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
		if(self.use_pixelshuffle == True):
			out = self.pixelshuffle(out)
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

		self.deconv = nn.ConvTranspose2d(ni, no, ks, stride, pad, output_padding = output_pad, bias = False)

		if(self.use_bn == True):
			if(self.norm_type == 'batchnorm'):
				self.bn = nn.BatchNorm2d(no)
			elif(self.norm_type == 'instancenorm'):
				self.bn = nn.InstanceNorm2d(no)

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
	def __init__(self, ic, oc, sz, nz = None, use_bn = True, use_sn = False, norm_type = 'instancenorm', use_norm_bottleneck = False, use_pixelshuffle = False):
		super(UNet_G, self).__init__()
		self.ic = ic
		self.oc = oc
		self.sz = sz
		self.nz = nz
		self.use_bn = use_bn
		self.use_sn = use_sn
		self.use_norm_btnk = use_norm_bottleneck
		self.use_pixelshuffle = use_pixelshuffle
		self.dims = {
			'16' : [64, 128, 256, 512],
			'32' : [64, 128, 256, 512, 512],
			'64' : [64, 128, 256, 512, 512, 512],
			'128' : [64, 128, 256, 512, 512, 512, 512],
			'256' : [64, 128, 256, 512, 512, 512, 512, 512],
			'512' : [64, 128, 256, 512, 512, 512, 512, 512, 512]
		}
		self.cur_dim = self.dims[str(sz)]
		self.num_convs = len(self.cur_dim)

		self.leaky_relu = nn.LeakyReLU(0.2, inplace = True)
		self.relu = nn.ReLU(inplace = True)

		self.enc_convs = nn.ModuleList([])
		if(self.nz == None):
			cur_block_ic = self.ic
		else:
			cur_block_ic = self.ic + self.nz

		for i, dim in enumerate(self.cur_dim):
			if(i == 0):
				self.enc_convs.append(ConvBlock(cur_block_ic, dim, 4, 2, 1, use_bn = False, use_sn = self.use_sn, activation_type = None))
			elif(i == len(self.cur_dim) - 1):
				self.enc_convs.append(ConvBlock(cur_block_ic, dim, 4, 2, 1, use_bn = self.use_norm_btnk, use_sn = self.use_sn, norm_type = norm_type, activation_type = None))
			else:
				self.enc_convs.append(ConvBlock(cur_block_ic, dim, 4, 2, 1, use_bn = self.use_bn, use_sn = self.use_sn, norm_type = norm_type, activation_type = None))
			cur_block_ic = dim

		self.dec_convs = nn.ModuleList([])
		cur_block_ic = self.cur_dim[-1]
		if(self.use_pixelshuffle):
			for i, dim in enumerate(list(reversed(self.cur_dim))[1:] + [self.oc]):
				if(i == 0):
					self.dec_convs.append(ConvBlock(cur_block_ic, dim, 3, 1, 1, use_bn = False, use_sn = self.use_sn, activation_type = None, use_pixelshuffle = True))
				elif(i == len(self.cur_dim) - 1):
					self.dec_convs.append(ConvBlock(cur_block_ic*2, self.oc, 3, 1, 1, use_bn = self.use_norm_btnk, use_sn = self.use_sn, norm_type = norm_type, activation_type = None, use_pixelshuffle = True))
				else:
					self.dec_convs.append(ConvBlock(cur_block_ic*2, dim, 3, 1, 1, use_bn = self.use_bn, use_sn = self.use_sn, norm_type = norm_type, activation_type = None, use_pixelshuffle = True))
				cur_block_ic = dim
		else:
			for i, dim in enumerate(list(reversed(self.cur_dim))[1:] + [self.oc]):
				if(i == 0):
					self.dec_convs.append(DeConvBlock(cur_block_ic, dim, 4, 2, 1, use_bn = False, use_sn = self.use_sn, activation_type = None))
				elif(i == len(self.cur_dim) - 1):
					self.dec_convs.append(DeConvBlock(cur_block_ic*2, self.oc, 4, 2, 1, use_bn = self.use_norm_btnk, use_sn = self.use_sn, norm_type = norm_type, activation_type = None))
				else:
					self.dec_convs.append(DeConvBlock(cur_block_ic*2, dim, 4, 2, 1, use_bn = self.use_bn, use_sn = self.use_sn, norm_type = norm_type, activation_type = None))
				cur_block_ic = dim

		self.tanh = nn.Tanh()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()
	
	def forward(self, x, z):
		ens = []
		if(z is None):
			out = x
		else:
			out = torch.cat([x, z.expand(-1, -1, x.shape[2], x.shape[3])], 1)

		for i, cur_enc in enumerate(self.enc_convs):
			if(i == 0):
				out = cur_enc(out)
			else:
				out = cur_enc(self.leaky_relu(out))
			ens.append(out)

		for i, cur_dec in enumerate(self.dec_convs):
			cur_enc = ens[self.num_convs - 1 - i]
			if(i == 0):
				out = cur_dec(self.relu(cur_enc))
			else:
				out = cur_dec(self.relu(torch.cat([out, cur_enc], 1)))
				
		del ens
		out = self.tanh(out)
		return out

