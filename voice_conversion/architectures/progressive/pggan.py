import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
from torch.nn.init import kaiming_normal_, calculate_gain

class EqualizedConv(nn.Module):
	def __init__(self, ni, no, ks, stride, pad, use_bias):
		super(EqualizedConv, self).__init__()
		self.ni = ni
		self.no = no
		self.ks = ks
		self.stride = stride
		self.pad = pad
		self.use_bias = use_bias

		self.weight = nn.Parameter(nn.init.normal_(
			torch.empty(self.no, self.ni, self.ks, self.ks)
		))
		if(self.use_bias):
			self.bias = nn.Parameter(torch.FloatTensor(self.no).fill_(0))

		self.scale = math.sqrt(2 / (self.ni * self.ks * self.ks))

	def forward(self, x):
		out = F.conv2d(input = x, weight = self.weight * self.scale, bias = self.bias,
					   stride = self.stride, padding = self.pad)
		return out

class ScaledConvBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad, act = 'relu', use_bias = True, use_equalized_lr = True, use_pixelnorm = True, only_conv = False):
		super(ScaledConvBlock, self).__init__()
		self.ni = ni
		self.no = no
		self.ks = ks
		self.stride = stride
		self.pad = pad
		self.act = act
		self.use_bias = use_bias
		self.use_equalized_lr = use_equalized_lr
		self.use_pixelnorm = use_pixelnorm
		self.only_conv = only_conv

		self.relu = nn.LeakyReLU(0.2, inplace = True)
		if(self.use_equalized_lr):
			'''
			self.conv = nn.Conv2d(ni, no, ks, stride, pad, bias = False)
			kaiming_normal_(self.conv.weight, a = calculate_gain('conv2d'))

			self.bias = torch.nn.Parameter(torch.FloatTensor(no).fill_(0))
			self.scale = (torch.mean(self.conv.weight.data ** 2)) ** 0.5
			self.conv.weight.data.copy_(self.conv.weight.data / self.scale)
			'''
			self.conv = EqualizedConv(ni, no, ks, stride, pad, use_bias = use_bias)
		
		else:
			self.conv = nn.Conv2d(ni, no, ks, stride, pad, bias = use_bias)

		if(self.use_pixelnorm):
			self.pixel_norm = PixelNorm()


	def forward(self, x):
		'''
		if(self.use_equalized_lr):
			out = self.conv(x * self.scale)
			out = out + self.bias.view(1, -1, 1, 1).expand_as(out)
		else:
			out = self.conv(x)
		'''
		out = self.conv(x)

		if(self.only_conv == False):
			if(self.act == 'relu'):
				out = self.relu(out)
			if(self.use_pixelnorm):
				out = self.pixel_norm(out)

		return out

class SPADE(nn.Module):
	def __init__(self, ic_1, ic_2):
		super(SPADE, self).__init__()
		self.ic_1 = ic_1	# channel number for x
		self.ic_2 = ic_2	# channel number for con
		self.bn = nn.BatchNorm2d(self.ic_1, affine = False)

		self.conv = ScaledConvBlock(self.ic_2, 128, 3, 1, 1, only_conv = True)
		self.gamma_conv = ScaledConvBlock(128, self.ic_1, 3, 1, 1, only_conv = True)
		self.beta_conv = ScaledConvBlock(128, self.ic_1, 3, 1, 1, only_conv = True)

	def forward(self, x, con):	# input shape = output shape
		normalized = self.bn(x)

		r_con = F.adaptive_avg_pool2d(con, (x.shape[2], x.shape[3]))
		r_con = self.conv(r_con)
		gamma = self.gamma_conv(r_con)
		beta = self.beta_conv(r_con)

		out = gamma * normalized + beta
		return out
		
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

# Progressive Architectures

class Minibatch_Stddev(nn.Module):
	def __init__(self):
		super(Minibatch_Stddev, self).__init__()

	def forward(self, x):
		stddev = torch.sqrt(torch.mean((x - torch.mean(x, dim = 0, keepdim = True))**2, dim = 0, keepdim = True) + 1e-8)
		stddev_mean = torch.mean(stddev, dim = 1, keepdim = True)
		stddev_mean = stddev_mean.expand((x.size(0), 1, x.size(2), x.size(3)))

		return torch.cat([x, stddev_mean], dim = 1)

class PixelNorm(nn.Module):
	def __init__(self):
		super(PixelNorm, self).__init__()

	def forward(self, x):
		out = x / torch.sqrt(torch.mean(x**2, dim = 1, keepdim = True) + 1e-8)
		return out

class PGGAN_G(nn.Module):
	def __init__(self, ic, oc, sz, nz, use_pixelnorm = False, use_equalized_lr = False, use_tanh = True):
		super(PGGAN_G, self).__init__()
		self.ic = ic
		self.oc = oc
		self.sz = sz
		self.nz = nz
		self.ngfs = {
			'8': [512, 512],
			'16': [512, 512, 512],
			'32': [512, 512, 512, 512],
			'64': [512, 512, 512, 512, 256],
			'128': [512, 512, 512, 512, 256, 128],
			'256': [512, 512, 512, 512, 256, 128, 64],
			'512': [512, 512, 512, 512, 256, 128, 64, 32],
			'1024': [512, 512, 512, 512, 256, 128, 64, 32, 16]
		}

		self.cur_ngf = self.ngfs[str(sz)]
		self.leakyrelu = nn.LeakyReLU(0.2, inplace = True)

		if(self.nz == None):
			default_nz_size = 8
			nz = default_nz_size
			self.constant = torch.nn.Parameter(torch.ones((1, nz, 1, 1)))
			self.constant.requires_grad = True
		else:
			self.constant = None

		# create blocks list
		prev_dim = self.cur_ngf[0]
		cur_block = ScaledConvBlock(nz, prev_dim, 4, 1, 3, 'relu', True, use_equalized_lr, use_pixelnorm)
		cur_spade_block = SPADE(prev_dim, self.ic)
		self.blocks = nn.ModuleList([cur_block])
		self.spade_blocks = nn.ModuleList([cur_spade_block])
		for dim in self.cur_ngf[1:]:
			cur_block = ScaledConvBlock(prev_dim, dim, 3, 1, 1, 'relu', True, use_equalized_lr, use_pixelnorm)
			cur_spade_block = SPADE(dim, self.ic)
			prev_dim = dim
			self.blocks.append(cur_block)
			self.spade_blocks.append(cur_spade_block)

		# create to_blocks list
		self.to_blocks = nn.ModuleList([])
		for dim in self.cur_ngf:
			self.to_blocks.append(ScaledConvBlock(dim, oc, 1, 1, 0, None, True, use_equalized_lr, use_pixelnorm, only_conv = True))

		self.use_tanh = use_tanh
		self.tanh = nn.Tanh()
		self.upsample = UpSample()

	def forward(self, con, z, stage):
		stage += 3
		stage_int = int(stage)
		stage_type = (stage == stage_int)

		if(z is None):
			out = self.constant.expand(con.size(0), -1, -1, -1)
		else:
			out = z

		# Stablization Steps
		if(stage_type):
			for i in range(stage_int):
				out = self.blocks[i](out)
				out = self.spade_blocks[i](out, con)
				out = self.leakyrelu(out)
				out = self.upsample(out)
			out = self.blocks[stage_int](out)
			out = self.spade_blocks[stage_int](out, con)
			out = self.leakyrelu(out)
			out = self.to_blocks[stage_int](out)

		# Growing Steps
		else:
			p = stage - stage_int
			for i in range(stage_int+1):
				out = self.blocks[i](out)
				out = self.spade_blocks[i](out, con)
				out = self.leakyrelu(out)
				out = self.upsample(out)

			out_1 = self.to_blocks[stage_int](out)
			out_2 = self.blocks[stage_int+1](out)
			out_2 = self.spade_blocks[stage_int+1](out_2, con)
			out_2 = self.leakyrelu(out_2)
			out_2 = self.to_blocks[stage_int+1](out_2)
			out = out_1 * (1 - p) + out_2 * p

		if(self.use_tanh):	
			out = self.tanh(out)

		return out

class PGGAN_D(nn.Module):
	def __init__(self, sz, nc, use_sigmoid = True, use_pixelnorm = False, use_equalized_lr = False):
		super(PGGAN_D, self).__init__()
		self.sz = sz
		self.nc = nc
		self.sigmoid = nn.Sigmoid()
		self.use_sigmoid = use_sigmoid
		self.ndfs = {
			'8': [512, 512],
			'16': [512, 512, 512],
			'32': [512, 512, 512, 512],
			'64': [512, 512, 512, 512, 256],
			'128': [512, 512, 512, 512, 256, 128],
			'256': [512, 512, 512, 512, 256, 128, 64],
			'512': [512, 512, 512, 512, 256, 128, 64, 32],
			'1024': [512, 512, 512, 512, 256, 128, 64, 32, 16]
		}

		self.cur_ndf = self.ndfs[str(sz)]

		# create blocks list
		prev_dim = self.cur_ndf[0]
		cur_block = nn.Sequential(
			Minibatch_Stddev(),
			ScaledConvBlock(prev_dim+1, prev_dim, 3, 1, 1, 'relu', True, use_equalized_lr, use_pixelnorm),
			ScaledConvBlock(prev_dim, prev_dim, 4, 1, 0, 'relu', True, use_equalized_lr, use_pixelnorm)
		)
		self.blocks = nn.ModuleList([cur_block])
		for dim in self.cur_ndf[1:]:
			cur_block = nn.Sequential(
				ScaledConvBlock(dim, dim, 3, 1, 1, 'relu', True, use_equalized_lr, use_pixelnorm),
				ScaledConvBlock(dim, prev_dim, 3, 1, 1, 'relu', True, use_equalized_lr, use_pixelnorm)
			)
			prev_dim = dim
			self.blocks.append(cur_block)

		# create from_blocks list
		self.from_blocks = nn.ModuleList([])
		for dim in self.cur_ndf:
			self.from_blocks.append(ScaledConvBlock(nc, dim, 1, 1, 0, None, True, use_equalized_lr, use_pixelnorm, only_conv = True))

		self.linear = nn.Linear(self.cur_ndf[0], 1)
		self.downsample = DownSample()

	def forward(self, x, stage):
		stage += 3
		stage_int = int(stage)
		stage_type = (stage == stage_int)
		out = x

		# Stablization Steps
		if(stage_type):
			out = self.from_blocks[stage_int](out)
			for i in range(stage_int):
				out = self.blocks[stage_int - i](out)
				out = self.downsample(out)
			out = self.blocks[0](out)
			out = self.linear(out.view(out.shape[0], -1))
			out = out.view(out.shape[0], 1, 1, 1)

		# Growing Steps
		else:
			p = stage - stage_int
			out_1 = self.downsample(out)
			out_1 = self.from_blocks[stage_int](out_1)

			out_2 = self.from_blocks[stage_int+1](out)
			out_2 = self.blocks[stage_int+1](out_2)
			out_2 = self.downsample(out_2)

			out = out_1 * (1 - p) + out_2 * p

			for i in range(stage_int):
				out = self.blocks[stage_int - i](out)
				out = self.downsample(out)
			out = self.blocks[0](out)
			out = self.linear(out.view(out.shape[0], -1))
			out = out.view(out.shape[0], 1, 1, 1)

		if(self.use_sigmoid):
			out = self.sigmoid(out)

		return out

