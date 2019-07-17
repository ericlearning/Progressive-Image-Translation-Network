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

def conv3x3(ic, oc, use_sn = True, use_reflection = True, use_eq = False):
	if(use_reflection):
		reflection = nn.ReflectionPad2d(1)
		if(use_eq):
			conv = EqualizedConv(ic, oc, 3, 1, 0, use_bias = True)
		else:
			conv = nn.Conv2d(ic, oc, 3, 1, 0, bias = True)
	else:
		reflection = Nothing()
		if(use_eq):
			conv = EqualizedConv(ic, oc, 3, 1, 1, use_bias = True)
		else:
			conv = nn.Conv2d(ic, oc, 3, 1, 1, bias = True)

	if(use_sn):
		conv = SpectralNorm(conv)

	out_list = [reflection, conv]
	return nn.Sequential(*out_list)

class EqualizedConv(nn.Module):
	def __init__(self, ni, no, ks, stride, pad, use_bias):
		super(EqualizedConv, self).__init__()
		self.ni = ni
		self.no = no
		self.ks = ks
		self.stride = stride
		self.pad = pad

		self.weight = nn.Parameter(nn.init.normal_(
			torch.empty(self.no, self.ni, self.ks, self.ks)
		))
		self.bias = None
		if(use_bias):
			self.bias = nn.Parameter(torch.FloatTensor(self.no).fill_(0))

		self.scale = math.sqrt(2 / (self.ni * self.ks * self.ks))

	def forward(self, x):
		out = F.conv2d(input = x, weight = self.weight * self.scale, bias = self.bias,
					   stride = self.stride, padding = self.pad)
		return out

class SPADE(nn.Module):
	def __init__(self, ic_1, ic_2, use_sn = True, use_reflection = True, use_eq = False):
		super(SPADE, self).__init__()
		self.ic_1 = ic_1	# channel number for x
		self.ic_2 = ic_2	# channel number for con
		self.bn = nn.BatchNorm2d(self.ic_1, affine = False)

		self.conv = conv3x3(self.ic_2, 128, use_sn, use_reflection, use_eq)
		self.gamma_conv = conv3x3(128, self.ic_1, use_sn, use_reflection, use_eq)
		self.beta_conv = conv3x3(128, self.ic_1, use_sn, use_reflection, use_eq)

	def forward(self, x, con):	# input shape = output shape
		normalized = self.bn(x)

		r_con = F.adaptive_avg_pool2d(con, (x.shape[2], x.shape[3]))
		r_con = self.conv(r_con)
		gamma = self.gamma_conv(r_con)
		beta = self.beta_conv(r_con)

		out = gamma * normalized + beta
		return out

class SPADE_ResBlk(nn.Module):
	def __init__(self, ic, oc, channel_c, use_sn = True, use_reflection = True, use_eq = False):
		super(SPADE_ResBlk, self).__init__()
		self.ic = ic
		self.oc = oc
		self.channel_c = channel_c

		self.spade_1 = SPADE(ic, channel_c, use_sn, use_reflection, use_eq)
		self.spade_2 = SPADE(oc, channel_c, use_sn, use_reflection, use_eq)
		self.spade_skip = SPADE(ic, channel_c, use_sn, use_reflection, use_eq)

		self.conv_1 = conv3x3(ic, oc, use_sn, use_reflection, use_eq)
		self.conv_2 = conv3x3(oc, oc, use_sn, use_reflection, use_eq)
		self.conv_skip = conv3x3(ic, oc, use_sn, use_reflection, use_eq)

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
	def __init__(self, ic, oc, sz, nz, use_sn = True, use_reflection = True, use_eq = False):
		super(SPADE_G_Progressive, self).__init__()
		self.ic = ic
		self.oc = oc
		self.sz = sz
		self.nz = nz

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
			self.blocks.append(SPADE_ResBlk(prev_res, res, ic, use_sn, use_reflection, use_eq))
			prev_res = res

		self.convs = nn.ModuleList([])
		for i in range(int(math.log(self.sz // 32, 2))+1):
			conv = conv3x3(self.res_num[i+2], oc, use_sn, use_reflection, use_eq)
			self.convs.append(conv)

		self.tanh = nn.Tanh()
		self.upsample = UpSample()

		if(self.nz == None):
			self.constant = torch.nn.Parameter(torch.ones((1, 1024, 4, 4)))
			self.constant.requires_grad = True
		else:
			self.linear = nn.Linear(nz, 4*4*1024)

		for m in self.modules():
			if(isinstance(m, nn.Conv2d)):
				nn.init.xavier_normal_(m.weight)

	def freeze(self, model, choice):
		for child in model.children():
			for param in child.parameters():
				param.requires_grad = not choice

	def freeze_parameter(self, param, choice):
		param.requires_grad = not choice

	def forward(self, con, z, stage):
		stage_int = int(stage)
		stage_type = (stage == stage_int)
		if(not stage_type):
			stage_int += 1

		if(stage_type):
			if(z is None):
				self.freeze_parameter(self.constant, False)
			else:
				self.freeze(self.linear, False)
			for i in range(0, stage_int+3):
				self.freeze(self.blocks[i], False)
			for i in range(stage_int+3, len(self.blocks)):
				self.freeze(self.blocks[i], True)
			self.freeze(self.convs[stage_int], False)

		else:
			if(z is None):
				self.freeze_parameter(self.constant, True)
			else:
				self.freeze(self.linear, True)
			for i in range(0, stage_int+2):
				self.freeze(self.blocks[i], True)
			self.freeze(self.blocks[stage_int+2], False)
			for i in range(stage_int+3, len(self.blocks)):
				self.freeze(self.blocks[i], True)
			self.freeze(self.convs[stage_int], False)


		if(z is None):
			out = self.constant.expand(con.size(0), -1, -1, -1)
		else:
			out = self.linear(z.view(-1, self.nz))
			out = out.view(-1, 1024, 4, 4)

		for i in range(stage_int + 3):
			out = self.upsample(self.blocks[i](out, con))
		out = self.convs[stage_int](out)
		out = self.tanh(out)
		
		return out






