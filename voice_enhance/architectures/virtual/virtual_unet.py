import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
from architectures.virtual.virtual_batchnorm import VirtualBatchnorm1D
# from virtual_batchnorm import VirtualBatchnorm1D

def get_norm(norm_type, size):
	if(norm_type == 'batchnorm'):
		return VirtualBatchnorm1D(size)
	elif(norm_type == 'instancenorm'):
		return nn.InstanceNorm1d(size)

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
				self.bn = VirtualBatchnorm1D(no)
			elif(self.norm_type == 'instancenorm'):
				self.bn = nn.InstanceNorm1d(no)
		else:
			self.bn = Nothing()

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

	def forward(self, x, ref_mean_in = None, ref_var_in = None):
		ref_mean, ref_var = None, None
		out = x
		if(self.pad_type == 'Reflection'):
			out = self.reflection(out)
		out = self.conv(out)

		if(self.use_bn == True):
			if(ref_mean_in is None or ref_var_in is None):
				ref_mean, ref_var, out = self.bn(out, ref_mean_in, ref_var_in)
			else:
				out = self.bn(out, ref_mean_in, ref_var_in)
		out = self.act(out)
		return ref_mean, ref_var, out


class DeConvBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad = None, pad_type = 'Zero', output_pad = 0, use_bn = True, use_sn = False, norm_type = 'batchnorm', activation_type = 'leakyrelu'):
		super(DeConvBlock, self).__init__()
		self.use_bn = use_bn
		self.use_sn = use_sn
		self.norm_type = norm_type
		self.pad_type = pad_type

		if(pad is None):
			pad = ks // 2 // stride

		if(self.pad_type == 'Zero'):
			self.deconv = nn.ConvTranspose1d(ni, no, ks, stride, pad, output_padding = output_pad, bias = False)
		elif(self.pad_type == 'Reflection'):
			self.deconv = nn.ConvTranspose1d(ni, no, ks, stride, 0, output_padding = output_pad, bias = False)
			self.reflection = nn.ReflectionPad1d(pad)
		
		if(self.use_bn == True):
			if(self.norm_type == 'batchnorm'):
				self.bn = VirtualBatchnorm1D(no)
			elif(self.norm_type == 'instancenorm'):
				self.bn = nn.InstanceNorm1d(no)
		else:
			self.bn = Nothing()

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

	def forward(self, x, ref_mean_in = None, ref_var_in = None):
		ref_mean, ref_var = None, None
		out = x
		if(self.pad_type == 'Reflection'):
			out = self.reflection(out)
		out = self.deconv(out)

		if(self.use_bn == True):
			if(ref_mean_in is None or ref_var_in is None):
				ref_mean, ref_var, out = self.bn(out, ref_mean_in, ref_var_in)
			else:
				out = self.bn(out, ref_mean_in, ref_var_in)
		out = self.act(out)
		return ref_mean, ref_var, out
		
def receptive_calculator(input_size, ks, stride, pad):
	return int((input_size - ks + 2 * pad) / stride + 1)

def inverse_receptive_calculator(output_size, ks, stride, pad):
	return ((output_size - 1) * stride) + ks

class Virtual_UNet_G(nn.Module):
	def __init__(self, ic = 1, oc = 1, use_sn = False, norm_type = 'instancenorm'):
		super(Virtual_UNet_G, self).__init__()
		self.ref = None

		self.ic = ic
		self.oc = oc
		self.use_sn = use_sn

		self.leaky_relu = nn.LeakyReLU(0.2, inplace = True)
		self.relu = nn.ReLU(inplace = True)

		self.cur_dim = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]

		self.enc_convs = nn.ModuleList([])
		cur_block_ic = 1
		for i, dim in enumerate(self.cur_dim):
			if(i == 0):
				block = ConvBlock(self.ic, self.cur_dim[0], 32, 2, 15, pad_type = 'Zero', use_bn = False, use_sn = self.use_sn, activation_type = None)
			elif(i == len(self.cur_dim) - 1):
				block = ConvBlock(self.cur_dim[i-1], self.cur_dim[i], 32, 2, 15, pad_type = 'Zero', use_bn = False, use_sn = self.use_sn, activation_type = None)
			else:
				block = ConvBlock(self.cur_dim[i-1], self.cur_dim[i], 32, 2, 15, pad_type = 'Zero', use_bn = True, norm_type = norm_type, use_sn = self.use_sn, activation_type = None)
			self.enc_convs.append(block)

		self.dec_convs = nn.ModuleList([])
		cur_block_ic = self.cur_dim[-1]
		for i, dim in enumerate(list(reversed(self.cur_dim))[1:] + [self.oc]):
			if(i == 0):
				de_block = DeConvBlock(cur_block_ic, dim, 32, 2, 15, pad_type = 'Zero', use_bn = False, use_sn = self.use_sn, activation_type = None)
			elif(i == len(self.cur_dim) - 1):
				de_block = DeConvBlock(cur_block_ic*2, self.oc, 32, 2, 15, pad_type = 'Zero', use_bn = False, use_sn = self.use_sn, activation_type = None)
			else:
				de_block = DeConvBlock(cur_block_ic*2, dim, 32, 2, 15, pad_type = 'Zero', use_bn = True, use_sn = self.use_sn, norm_type = norm_type, activation_type = None)
			cur_block_ic = dim
			self.dec_convs.append(de_block)

		self.prelus = nn.ModuleList([])
		for i in range(len(self.cur_dim) * 2 - 1):
			self.prelus.append(nn.PReLU())

		self.tanh = nn.Tanh()

		for m in self.modules():
			if(isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d)):
				nn.init.xavier_normal_(m.weight)

	def forward_ref(self, ref):
		ens = []
		out = ref

		ref_means = []
		ref_vars = []

		with torch.no_grad():
			cnt2 = 0
			for i, cur_enc in enumerate(self.enc_convs):
				if(i == 0):									# no bn
					_, _, out = cur_enc(out)
				elif(i == len(self.cur_dim) - 1):			# no bn
					_, _, out = cur_enc(self.prelus[cnt2](out))
				else:										# yes bn
					ref_mean, ref_var, out = cur_enc(self.prelus[cnt2](out), None, None)
					ref_means.append(ref_mean)
					ref_vars.append(ref_var)
				cnt2 += 1
				ens.append(out)

			for i, cur_dec in enumerate(self.dec_convs):
				cur_enc = ens[len(self.cur_dim) - 1 - i]
				if(i == 0):									# no bn
					_, _, out = cur_dec(self.prelus[cnt2](cur_enc))
				elif(i == len(self.cur_dim) - 1):			# no bn
					_, _, out = cur_dec(self.prelus[cnt2](torch.cat([out, cur_enc], 1)))
				else:										# yes bn
					ref_mean, ref_var, out = cur_dec(self.prelus[cnt2](torch.cat([out, cur_enc], 1)), None, None)
					ref_means.append(ref_mean)
					ref_vars.append(ref_var)
				cnt2 += 1
					
			del ens
			out = self.tanh(out)
			
		return ref_means, ref_vars, out

	def forward_normal(self, x, ref_means, ref_vars):
		ens = []
		out = x
		cnt, cnt2 = 0, 0

		for i, cur_enc in enumerate(self.enc_convs):
			if(i == 0):									# no bn
				_, _, out = cur_enc(out)
			elif(i == len(self.cur_dim) - 1):			# no bn
				_, _, out = cur_enc(self.prelus[cnt2](out))
			else:										# yes bn
				ref_mean, ref_var = ref_means[cnt], ref_vars[cnt]
				cnt += 1
				_, _, out = cur_enc(self.prelus[cnt2](out), ref_mean, ref_var)
			ens.append(out)
			cnt2 += 1

		for i, cur_dec in enumerate(self.dec_convs):
			cur_enc = ens[len(self.cur_dim) - 1 - i]
			if(i == 0):									# no bn
				_, _, out = cur_dec(self.prelus[cnt2](cur_enc))
			elif(i == len(self.cur_dim) - 1):			# no bn
				_, _, out = cur_dec(self.prelus[cnt2](torch.cat([out, cur_enc], 1)))
			else:
				ref_mean, ref_var = ref_means[cnt], ref_vars[cnt]
				cnt += 1										# yes bn
				_, _, out = cur_dec(self.prelus[cnt2](torch.cat([out, cur_enc], 1)), ref_mean, ref_var)
			cnt2 += 1
				
		del ens
		out = self.tanh(out)
		return out

	def forward(self, x):
		if((self.training) and (self.ref is None)):
			self.ref = x.detach().clone()
		ref_means, ref_vars, _ = self.forward_ref(self.ref)
		out = self.forward_normal(x, ref_means, ref_vars)

		return out