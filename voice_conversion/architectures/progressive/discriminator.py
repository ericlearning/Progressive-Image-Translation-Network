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

class DownSample(nn.Module):
	def __init__(self):
		super(DownSample, self).__init__()

	def forward(self, x):
		return F.avg_pool2d(x, 2)

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

class PatchGan_D_70x70_One_Input(nn.Module):
	def __init__(self, ic, sz, use_sigmoid = True, norm_type = 'instancenorm', use_sn = False):
		super(PatchGan_D_70x70_One_Input, self).__init__()
		self.ic = ic
		self.sz = sz
		self.use_sn = use_sn
		self.use_sigmoid = use_sigmoid

		self.res_nums = {
			'32' : [512],
			'64' : [512, 256],
			'128' : [512, 256, 128],
			'256' : [512, 256, 128, 64],
			'512' : [512, 256, 128, 64, 32]
		}
		self.res_num = self.res_nums[str(self.sz)]

		self.from_blocks = nn.ModuleList([])
		for i, res in enumerate(self.res_num):
			if(i == 0):
				self.from_blocks.append(ConvBlock(self.ic, res, 4, 1, 1, use_bn = False, activation_type = 'leakyrelu', use_sn = self.use_sn))
			else:
				self.from_blocks.append(ConvBlock(self.ic, res, 4, 2, 1, use_bn = False, activation_type = 'leakyrelu', use_sn = self.use_sn))

		self.blocks = nn.ModuleList([])
		for i in range(len(self.res_num)-1):
			if(i == 0):
				self.blocks.append(ConvBlock(self.res_num[i+1], self.res_num[i], 4, 1, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu', use_sn = self.use_sn))
			else:
				self.blocks.append(ConvBlock(self.res_num[i+1], self.res_num[i], 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu', use_sn = self.use_sn))

		self.last_block = ConvBlock(self.res_num[0], 1, 4, 1, 1, use_bn = False, activation_type = None, use_sn = self.use_sn)

		self.sigmoid = nn.Sigmoid()
		self.nothing = Nothing()
		self.ds = DownSample()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x, stage):
		stage_int = int(stage)
		stage_type = (stage == stage_int)
		out = x

		if(stage_type):
			out = self.from_blocks[stage_int](out)
			for i in range(stage_int):
				out = self.blocks[stage_int - i - 1](out)
			out = self.last_block(out)

		else:
			p = stage - stage_int

			out_1 = self.ds(out)
			out_1 = self.from_blocks[stage_int](out_1)

			out_2 = self.from_blocks[stage_int+1](out)
			out_2 = self.blocks[stage_int](out_2)

			out = out_1 * (1-p) + out_2 * p
			for i in range(stage_int):
				out = self.blocks[stage_int - i - 1](out)
			out = self.last_block(out)

		if(self.use_sigmoid == True):
			out = self.sigmoid(out)
		else:
			out = self.nothing(out)

		return out
