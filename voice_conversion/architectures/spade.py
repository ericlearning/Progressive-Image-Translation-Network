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

class SPADE(nn.Module):
	def __init__(self, ic_1, ic_2):
		self.ic_1 = ic_1	# channel number for x
		self.ic_2 = ic_2	# channel number for con
		self.k = k
		self.bn = nn.BatchNorm2d(self.ic_1, affine = False)
		self.conv = nn.Conv2d(self.ic_2, 128, 3, 1, 1, bias = True)
		self.relu = nn.ReLU(inplace = True)
		self.gamma_conv = nn.Conv2d(128, self.ic_1, 3, 1, 1, bias = True)
		self.beta_conv = nn.Conv2d(128, self.ic_1, 3, 1, 1, bias = True)

	def forward(self, x, con):	# input shape = output shape
		normalized = self.bn(x)

		r_con = F.avg_pool2d(con, (x.shape[2], x.shape[3]))
		r_con = self.conv(r_con)
		gamma = self.gamma_conv(r_con)
		beta = self.beta_conv(r_con)

		out = gamma * mean + beta
		return out

class SPADE_ResBlk(nn.Module):
	def __init__(self, ic, oc, channel_c):
		self.ic = ic
		self.oc = oc
		self.channel_c = channel_c

		self.spade_1 = SPADE(ic, channel_c)
		self.spade_2 = SPADE(oc, channel_c)
		self.spade_skip = SPADE(ic, channel_c)

		self.conv_1 = nn.Conv2d(ic, oc, 3, 1, 1, bias = True)
		self.conv_2 = nn.Conv2d(oc, oc, 3, 1, 1, bias = True)
		self.conv_skip = nn.Conv2d(ic, oc, 3, 1, 1, bias = True)
		self.relu = nn.ReLU(inplace = True)

	def forward(self, x, con):
		out = self.spade_1(x, con)
		out = self.relu(out)
		out = self.conv1(out)
		out = self.spade_2(out, con)
		out = self.relu(out)
		out = self.conv2(out)

		skip_out = self.spade_skip(x)
		skip_out = self.relu(skip_out)
		skip_out = self.conv_skip(skip_out)

		return out + skip_out

class SPADE_G(nn.Module):
	def __init__(self, ic, oc, sz, nz):
		self.ic = ic
		self.oc = oc
		self.sz = sz
		self.nz = nz

		self.linear = nn.Linear(nz, 4*4*1024)
		self.res_nums = {
			'16' : [1024, 1024]
			'32' : [1024, 1024, 1024]
			'64' : [1024, 1024, 1024, 512]
			'128' : [1024, 1024, 1024, 512, 256]
			'256' : [1024, 1024, 1024, 512, 256, 128]
			'512' : [1024, 1024, 1024, 512, 256, 128, 64]
		}
		self.res_num = self.res_num[str(sz)]

		prev_res = 1024
		self.blocks = nn.ModuleList([])
		for res in self.res_num:
			self.blocks.append(SPADE_ResBlk(prev_res, res, ic))
			prev_res = res

		self.conv = nn.Conv2d(prev_res, oc, 3, 1, 1, bias = True)
		self.tanh = nn.Tanh()

	def forward(self, con, z):
		out = self.linear(z.view(-1, self.nz))
		out = out.view(-1, 1024, 4, 4)

		for block in self.blocks:
			out = block(out)

		out = self.conv(out)
		out = self.tanh(out)
		
		return out






