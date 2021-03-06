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
	def __init__(self, ni, no, ks, stride, pad = None, pad_type = 'Zero', use_bn = True, use_pixelshuffle = False, norm_type = 'batchnorm', activation_type = 'leakyrelu'):
		super(ConvBlock, self).__init__()
		self.use_bn = use_bn
		self.use_pixelshuffle = use_pixelshuffle
		self.norm_type = norm_type
		self.pad_type = pad_type

		if(pad == None):
			pad = ks // 2 // stride

		if(use_pixelshuffle):
			if(self.pad_type == 'Zero'):
				self.conv = nn.Conv2d(ni, no * 2 * 2, ks, stride, pad, bias = False)
			elif(self.pad_type == 'Reflection'):
				self.conv = nn.Conv2d(ni, no * 2 * 2, ks, stride, 0, bias = False)
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
			elif(self.norm_type == 'spectralnorm'):
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
		if(self.pad_type == 'Reflection'):
			x = self.reflection(x)
		out = self.conv(x)
		if(self.use_pixelshuffle == True):
			out = self.pixelshuffle(out)
		if(self.use_bn == True and self.norm_type != 'spectralnorm'):
			out = self.bn(out)
		out = self.act(out)
		return out

class DeConvBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad = None, output_pad = 0, use_bn = True, norm_type = 'batchnorm', activation_type = 'leakyrelu'):
		super(DeConvBlock, self).__init__()
		self.use_bn = use_bn
		self.norm_type = norm_type

		if(pad is None):
			pad = ks // 2 // stride

		self.deconv = nn.ConvTranspose2d(ni, no, ks, stride, pad, output_padding = output_pad, bias = False)

		if(self.use_bn == True):
			if(self.norm_type == 'batchnorm'):
				self.bn = nn.BatchNorm2d(no)
			elif(self.norm_type == 'instancenorm'):
				self.bn = nn.InstanceNorm2d(no)
			elif(self.norm_type == 'spectralnorm'):
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
		if(self.use_bn == True and self.norm_type != 'spectralnorm'):
			out = self.bn(out)
		out = self.act(out)
		return out

class PatchGan_D_70x70(nn.Module):
	def __init__(self, ic_1, ic_2, use_sigmoid = True, norm_type = 'instancenorm', return_feature = False):
		super(PatchGan_D_70x70, self).__init__()
		self.ic_1 = ic_1
		self.ic_2 = ic_2
		self.use_sigmoid = use_sigmoid
		self.conv1 = ConvBlock(self.ic_1 + self.ic_2, 64, 4, 2, 1, use_bn = False, activation_type = 'leakyrelu')
		self.conv2 = ConvBlock(64, 128, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv3 = ConvBlock(128, 256, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv4 = ConvBlock(256, 512, 4, 1, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv5 = nn.Conv2d(512, 1, 4, 1, 1, bias = False)
		self.sigmoid = nn.Sigmoid()
		self.nothing = Nothing()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x1, x2, return_feature = False):
		out = torch.cat([x1, x2], 1)
		# (bs, ic_1+ic_2, 256, 256)
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

		if(return_feature):
			return out, [out1, out2, out3, out4, out5]
		else:
			return out
		
def receptive_calculator(input_size, ks, stride, pad):
	return int((input_size - ks + 2 * pad) / stride + 1)

def inverse_receptive_calculator(output_size, ks, stride, pad):
	return ((output_size - 1) * stride) + ks

class UpSample(nn.Module):
	def __init__(self):
		super(UpSample, self).__init__()

	def forward(self, x):
		return F.interpolate(x, None, 2, 'bilinear', align_corners=True)

class DownSample(nn.Module):
	def __init__(self):
		super(DownSample, self).__init__()

	def forward(self, x):
		return F.avg_pool2d(x, 2)

class UNet_G(nn.Module):
	def __init__(self, ic, oc, sz, nz = None, norm_type = 'instancenorm'):
		super(UNet_G, self).__init__()
		self.ic = ic
		self.oc = oc
		self.sz = sz
		self.nz = nz
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

		self.lc = nn.ModuleList([])
		for i in range(self.num_convs-1):
			if(i == self.num_convs-2):
				self.lc.append(ConvBlock(self.cur_dim[i], self.cur_dim[i+1], 4, 2, 1, use_bn = False, activation_type = None))
			else:
				self.lc.append(ConvBlock(self.cur_dim[i], self.cur_dim[i+1], 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = None))

		self.lp = nn.ModuleList([])
		for i in range(self.num_convs-1):
			if(i == self.num_convs-2):
				self.lp.append(DeConvBlock(self.cur_dim[i+1], self.cur_dim[i], 4, 2, 1, use_bn = False, activation_type = None))
			else:
				self.lp.append(DeConvBlock(self.cur_dim[i+1]*2, self.cur_dim[i], 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = None))

		self.from_convs = nn.ModuleList([])
		for i in range(self.num_convs-4):
			if(self.nz == None):
				self.from_convs.append(ConvBlock(self.ic, self.cur_dim[i], 1, 1, 0, use_bn = False, activation_type = None))
			else:
				self.from_convs.append(ConvBlock(self.ic + self.nz, self.cur_dim[i], 1, 1, 0, use_bn = False, activation_type = None))

		self.to_convs = nn.ModuleList([])
		for i in range(self.num_convs-4):
			self.to_convs.append(DeConvBlock(self.cur_dim[i], self.oc, 1, 1, 0, use_bn = False, activation_type = None))

		self.tanh = nn.Tanh()
		self.ds = DownSample()
		self.us = UpSample()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()
	
	def forward(self, x, stage, z = None):
		stage_int = int(stage)
		stage_type = (stage == stage_int)
		tmp = self.num_convs - 5
		p = stage - stage_int

		ens = []
		if(z is None):
			out = x
		else:
			out = torch.cat([x, z.expand(-1, -1, x.shape[2], x.shape[3])], 1)

		# Stablization Steps
		if(stage_type):
			tmp2 = tmp-stage_int
			out = self.from_convs[tmp2](out)
			for i in range(stage_int+4):
				out = self.lc[i+tmp2](self.leaky_relu(out))
				ens.append(out)

			tmp3 = self.num_convs - 2
			for i in range(stage_int+4):
				if(i == 0):
					out = self.lp[tmp3 - i](self.relu(out))
				else:
					out = self.lp[tmp3 - i](self.relu(torch.cat([ens[stage_int+5-i-2], out], 1)))
			out = self.to_convs[tmp2](self.relu(out))
			out = self.tanh(out)

			return out

		else:
			tmp2 = tmp-stage_int

			out1 = self.ds(out)
			out1 = self.from_convs[tmp2](out1)

			out2 = self.from_convs[tmp2-1](out)
			out2 = self.lc[tmp2-1](self.leaky_relu(out2))

			out = out1 * (1-p) + out2 * (p)

			for i in range(stage_int+4):
				out = self.lc[i+tmp2](self.leaky_relu(out))
				ens.append(out)

			tmp3 = self.num_convs - 2
			for i in range(stage_int+4):
				if(i == 0):
					out = self.lp[tmp3 - i](self.relu(out))
				else:
					out = self.lp[tmp3 - i](self.relu(torch.cat([ens[stage_int+5-i-2], out], 1)))

			out1_ = self.us(self.to_convs[tmp2](self.relu(out)))
			out2_ = self.lp[tmp2-1](self.relu(torch.cat([out2, out], 1)))
			out2_ = self.to_convs[tmp2-1](self.relu(out2_))

			out = out1_ * (1-p) + out2_ * (p)

			out = self.tanh(out)

			return out
