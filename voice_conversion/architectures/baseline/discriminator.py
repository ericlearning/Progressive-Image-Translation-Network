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

class PatchGan_D_70x70_One_Input(nn.Module):
	def __init__(self, ic, use_sigmoid = True, norm_type = 'instancenorm', use_sn = False):
		super(PatchGan_D_70x70_One_Input, self).__init__()
		self.ic = ic
		self.use_sn = use_sn
		self.use_sigmoid = use_sigmoid
		self.conv1 = ConvBlock(self.ic, 64, 4, 2, 1, use_bn = False, activation_type = 'leakyrelu', use_sn = self.use_sn)
		self.conv2 = ConvBlock(64, 128, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu', use_sn = self.use_sn)
		self.conv3 = ConvBlock(128, 256, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu', use_sn = self.use_sn)
		self.conv4 = ConvBlock(256, 512, 4, 1, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu', use_sn = self.use_sn)
		self.conv5 = ConvBlock(512, 1, 4, 1, 1, use_bn = False, activation_type = None, use_sn = self.use_sn)
		self.sigmoid = nn.Sigmoid()
		self.nothing = Nothing()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x, return_feature = False):
		out = x
		# (bs, ic, 256, 256)
		out1 = self.conv1(out)
		# (bs, 64, 128, 128)
		out2 = self.conv2(out1)
		# (bs, 128, 64, 64)
		out3 = self.conv3(out2)
		# (bs, 256, 32, 32)
		out4 = self.conv4(out3)
		# (bs, 512, 31, 31)
		out5 = self.conv5(out4)
		# (bs, 1, 30, 30)
		if(self.use_sigmoid == True):
			out = self.sigmoid(out5)
		else:
			out = self.nothing(out5)

		return out

class PatchGan_D_286x286_One_Input(nn.Module):
	def __init__(self, ic, use_sigmoid = True, norm_type = 'instancenorm', use_sn = False):
		super(PatchGan_D_286x286_One_Input, self).__init__()
		self.ic = ic
		self.use_sn = use_sn
		self.use_sigmoid = use_sigmoid
		self.conv1 = ConvBlock(self.ic, 64, 4, 2, 1, use_bn = False, activation_type = 'leakyrelu', use_sn = self.use_sn)
		self.conv2 = ConvBlock(64, 128, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu', use_sn = self.use_sn)
		self.conv3 = ConvBlock(128, 256, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu', use_sn = self.use_sn)
		self.conv4 = ConvBlock(256, 512, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu', use_sn = self.use_sn)
		self.conv5 = ConvBlock(512, 512, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu', use_sn = self.use_sn)
		self.conv6 = ConvBlock(512, 512, 4, 1, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu', use_sn = self.use_sn)
		self.conv7 = ConvBlock(512, 1, 4, 1, 1, use_bn = False, activation_type = None, use_sn = self.use_sn)
		self.sigmoid = nn.Sigmoid()
		self.nothing = Nothing()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x, return_feature = False):
		out = x
		# (bs, ic, 256, 256)
		out1 = self.conv1(out)
		# (bs, 64, 128, 128)
		out2 = self.conv2(out1)
		# (bs, 128, 64, 64)
		out3 = self.conv3(out2)
		# (bs, 256, 32, 32)
		out4 = self.conv4(out3)
		# (bs, 256, 16, 16)
		out5 = self.conv5(out4)
		# (bs, 256, 8, 8)
		out6 = self.conv6(out5)
		# (bs, 512, 7, 7)
		out7 = self.conv7(out6)
		# (bs, 1, 6, 6)
		if(self.use_sigmoid == True):
			out = self.sigmoid(out7)
		else:
			out = self.nothing(out7)

		return out
		