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

# Residual Block
class ResBlock(nn.Module):
	def __init__(self, ic, oc, norm_type = 'instancenorm', use_sn = False):
		super(ResBlock, self).__init__()
		self.ic = ic
		self.oc = oc
		self.norm_type = norm_type
		self.use_sn = use_sn

		self.relu = nn.ReLU(inplace = True)
		self.reflection_pad1 = nn.ReflectionPad2d(1)
		self.reflection_pad2 = nn.ReflectionPad2d(1)

		self.conv1 = nn.Conv2d(ic, oc, 3, 1, 0, bias = False)
		self.conv2 = nn.Conv2d(oc, oc, 3, 1, 0, bias = False)

		if(self.use_sn == True):
			self.conv1 = SpectralNorm(self.conv1)
			self.conv2 = SpectralNorm(self.conv2)
		else:
			if(self.norm_type == 'batchnorm'):
				self.bn1 = nn.BatchNorm2d(oc)
				self.bn2 = nn.BatchNorm2d(oc)

			elif(self.norm_type == 'instancenorm'):
				self.bn1 = nn.InstanceNorm2d(oc)
				self.bn2 = nn.InstanceNorm2d(oc)

	def forward(self, x):
		out = self.reflection_pad1(x)
		out = self.relu(self.bn1(self.conv1(out)))
		out = self.reflection_pad2(out)
		out = self.bn2(self.conv2(out))
		out = out + x
		return out

# ResNet Generator
class ResNet_G(nn.Module):
	def __init__(self, ic, oc, sz, nz = None, norm_type = 'instancenorm', use_sn = False):
		super(ResNet_G, self).__init__()
		self.ic = ic
		self.oc = oc
		self.sz = sz
		self.res_nums = {
			'16' : 2,
			'32' : 3,
			'64' : 4,
			'128' : 5,
			'256' : 6,
			'512' : 7
		}
		self.res_num = self.res_nums[str(sz)]
		self.nz = nz

		self.relu = nn.ReLU(inplace = True)

		self.reflection_pad1 = nn.ReflectionPad2d(3)
		self.reflection_pad2 = nn.ReflectionPad2d(3)

		if(self.nz == None):
			block_ic = self.ic
		else:
			block_ic = self.ic + self.nz

		self.conv = nn.Conv2d(block_ic, 64, 7, 1, 0)
		self.conv_block1 = ConvBlock(64, 128, 3, 2, pad = 1, use_bn = True, norm_type = norm_type, use_sn = use_sn)
		self.conv_block2 = ConvBlock(128, 256, 3, 2, pad = 1, use_bn = True, norm_type = norm_type, use_sn = use_sn)

		list_blocks = [ResBlock(256, 256, norm_type, use_sn = use_sn)] * self.res_num
		self.resblocks = nn.Sequential(*list_blocks)

		self.deconv_block1 = DeConvBlock(256, 128, 3, 2, pad = 1, output_pad = 1, use_bn = True, norm_type = norm_type, use_sn = use_sn)
		self.deconv_block2 = DeConvBlock(128, 64, 3, 2, pad = 1, output_pad = 1, use_bn = True, norm_type = norm_type, use_sn = use_sn)
		self.deconv = nn.Conv2d(64, oc, 7, 1, 0)

		if(use_sn):
			self.conv = SpectralNorm(self.conv)
			self.deconv = SpectralNorm(self.deconv)

		self.tanh = nn.Tanh()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x, z):
		if(z is None):
			out = x
		else:
			out = torch.cat([x, z.expand(-1, -1, x.shape[2], x.shape[3])], 1)

		# (bs, ic, sz, sz)
		out = self.reflection_pad1(out)
		# (bs, ic, sz+6, sz+6)
		out = self.conv(out)
		# (bs, 64, sz, sz)
		out = self.conv_block1(out)
		# (bs, 128, sz / 2, sz / 2)
		out = self.conv_block2(out)
		# (bs, 256, sz / 4, sz / 4)
		out = self.resblocks(out)
		# (bs, 256, sz / 4, sz / 4)
		out = self.deconv_block1(out)
		# (bs, 128, sz / 2, sz / 2)
		out = self.deconv_block2(out)
		# (bs, 64, sz, sz)
		out = self.reflection_pad2(out)
		# (bs, 64, sz+3, sz+3)
		out = self.deconv(out)
		# (bs, oc, sz, sz)
		out = self.tanh(out)
		# (bs, oc, sz, sz)
		return out