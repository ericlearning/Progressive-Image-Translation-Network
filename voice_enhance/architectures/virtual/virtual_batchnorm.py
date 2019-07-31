import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Custom_Batchnorm1D(nn.Module):
	def __init__(self, nc, momentum = 0.1, eps = 1e-8):
		super(Custom_Batchnorm1D, self).__init__()
		self.momentum = momentum
		self.eps = eps
		self.gamma = nn.Parameter(torch.Tensor(nc))
		self.beta = nn.Parameter(torch.Tensor(nc))

		self.r_mean = torch.Tensor(nc)
		self.r_var = torch.Tensor(nc)

		self.init_param(self.gamma, self.beta, self.r_mean, self.r_var)

	def init_param(self, gamma, beta, r_mean, r_var):
		r_mean.zero_()
		r_var.fill_(1)
		gamma.data.uniform_(0, 1)
		beta.data.zero_()

	def forward(self, x):
		bs, c, sz = x.shape
		out = x.permute(0, 2, 1).reshape(-1, c)

		if(self.training):
			mean = torch.mean(out, 0)
			var = torch.var(out, 0)
			out = ((out - mean) / (var + self.eps))
			
			self.r_mean = (1 - self.momentum) * self.r_mean + (self.momentum) * mean
			self.r_var = (1 - self.momentum) * self.r_var + (self.momentum) * var
		else:
			out = ((out - self.r_mean) / (self.r_var + self.eps))

		out = out * self.gamma + self.beta
		return out

class VirtualBatchnorm1D(nn.Module):
	def __init__(self, nc, eps = 1e-8):
		super(VirtualBatchnorm1D, self).__init__()
		self.eps = eps
		self.gamma = nn.Parameter(torch.Tensor(nc))
		self.beta = nn.Parameter(torch.Tensor(nc))

		self.init_param(self.gamma, self.beta)
		# reason for register_parameter : only way to assign None in a parameter
		# why assign None? Just to show that nothing is assigned in the beginning
		self.ref_mean = None
		self.ref_var = None

	def init_param(self, gamma, beta):
		gamma.data.uniform_(0, 1)
		beta.data.zero_()

	def forward(self, x, ref_mean = None, ref_var = None):
		bs, c, sz = x.shape
		out = x.permute(0, 2, 1).reshape(-1, c)

		if(ref_mean is None or ref_var is None):
			# reference time
			self.ref_mean = torch.mean(out, 0)
			self.ref_var = torch.var(out, 0)
			out = ((out - self.ref_mean) / (self.ref_var + self.eps))
			out = out * self.gamma + self.beta
			out = out.reshape(bs, sz, c).permute(0, 2, 1)

			# reference batch statistics, normalized output
			return self.ref_mean, self.ref_var, out

		else:
			# not reference time
			mean = torch.mean(out, 0)
			var = torch.var(out, 0)

			c_1 = 1.0 / (1 + bs)
			c_2 = 1.0 - c_1

			use_mean = mean * c_1 + ref_mean * c_2
			use_var = var * c_1 + ref_var * c_2

			out = ((out - use_mean) / (use_var + self.eps))
			out = out * self.gamma + self.beta
			out = out.reshape(bs, sz, c).permute(0, 2, 1)

			return out