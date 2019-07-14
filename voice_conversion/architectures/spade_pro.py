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

class UpSample(nn.Module):
	def __init__(self):
		super(UpSample, self).__init__()

	def forward(self, x):
		return F.interpolate(x, None, 2, 'bilinear', align_corners=True)

class SPADE(nn.Module):
	def __init__(self, ic_1, ic_2):
		super(SPADE, self).__init__()
		self.ic_1 = ic_1	# channel number for x
		self.ic_2 = ic_2	# channel number for con
		self.bn = nn.BatchNorm2d(self.ic_1, affine = False)
		self.conv = SpectralNorm(nn.Conv2d(self.ic_2, 128, 3, 1, 1, bias = True))
		self.gamma_conv = SpectralNorm(nn.Conv2d(128, self.ic_1, 3, 1, 1, bias = True))
		self.beta_conv = SpectralNorm(nn.Conv2d(128, self.ic_1, 3, 1, 1, bias = True))

	def forward(self, x, con):	# input shape = output shape
		normalized = self.bn(x)

		r_con = F.adaptive_avg_pool2d(con, (x.shape[2], x.shape[3]))
		r_con = self.conv(r_con)
		gamma = self.gamma_conv(r_con)
		beta = self.beta_conv(r_con)

		out = gamma * normalized + beta
		return out

class SPADE_ResBlk(nn.Module):
	def __init__(self, ic, oc, channel_c):
		super(SPADE_ResBlk, self).__init__()
		self.ic = ic
		self.oc = oc
		self.channel_c = channel_c

		self.spade_1 = SPADE(ic, channel_c)
		self.spade_2 = SPADE(oc, channel_c)
		self.spade_skip = SPADE(ic, channel_c)

		self.conv_1 = SpectralNorm(nn.Conv2d(ic, oc, 3, 1, 1, bias = True))
		self.conv_2 = SpectralNorm(nn.Conv2d(oc, oc, 3, 1, 1, bias = True))
		self.conv_skip = SpectralNorm(nn.Conv2d(ic, oc, 3, 1, 1, bias = True))
		self.relu = nn.ReLU(inplace = True)

	def forward(self, x, con):
		out = self.spade_1(x, con)
		out = self.relu(out)
		out = self.conv_1(out)
		out = self.spade_2(out, con)
		out = self.relu(out)
		out = self.conv_2(out)

		skip_out = self.spade_skip(x, con)
		skip_out = self.relu(skip_out)
		skip_out = self.conv_skip(skip_out)

		return out + skip_out

class SPADE_G_Progressive(nn.Module):
	def __init__(self, ic, oc, sz, nz):
		super(SPADE_G_Progressive, self).__init__()
		self.ic = ic
		self.oc = oc
		self.sz = sz
		self.nz = nz

		self.linear = nn.Linear(nz, 4*4*1024)
		self.res_nums = {
			'32' : [1024, 1024, 1024],
			'64' : [1024, 1024, 1024, 512],
			'128' : [1024, 1024, 1024, 512, 256],
			'256' : [1024, 1024, 1024, 512, 256, 128],
			'512' : [1024, 1024, 1024, 512, 256, 128, 64]
		}
		self.res_num = self.res_nums[str(sz)]

		prev_res = 1024
		self.blocks = nn.ModuleList([])
		for res in self.res_num:
			self.blocks.append(SPADE_ResBlk(prev_res, res, ic))
			prev_res = res

		self.convs = nn.ModuleList([])
		for i in range(int(math.log(self.sz // 32, 2))+1):
			conv = SpectralNorm(nn.Conv2d(self.res_num[i+2], oc, 3, 1, 1, bias = True))
			self.convs.append(conv)

		self.tanh = nn.Tanh()
		self.upsample = UpSample()

		if(self.nz == None):
			self.constant = torch.randn(1, 1024, 4, 4)

	def forward(self, con, z, stage):
		stage_int = int(stage)
		stage_type = (stage == stage_int)
		p = stage - stage_int

		if(z is None):
			out = self.constant.expand(con.size(0), -1, -1, -1)
		else:
			out = self.linear(z.view(-1, self.nz))
			out = out.view(-1, 1024, 4, 4)

		for i in range(stage_int + 3):
			out = self.upsample(self.blocks[i](out, con))

		if(stage_type):
			out = self.convs[stage_int](out)

		else:
			out1 = self.convs[stage_int](out)
			out1 = self.upsample(out1)

			out2 = self.blocks[stage_int + 3](out)
			out2 = self.upsample(out2)
			out2 = self.convs[stage_int+1](out2)

			out = out1 * (1-p) + out2 * p

		out = self.tanh(out)
		
		return out






