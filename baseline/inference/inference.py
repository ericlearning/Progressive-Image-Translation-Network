import copy
import torch
import torch.nn as nn
from torchvision import transforms
import os, cv2, sys
import numpy as np
sys.path.append('..')
from architectures.architecture import UNet_G
from utils.griffin_lim import *
from utils.utils import generate_noise
from PIL import Image

def get_image(img_dir, name_list, cnt, sz):
	image = cv2.imread(os.path.join(img_dir, name_list[cnt]))
	image = cv2.resize(image, (sz, sz))
	return image

def transform_image(image, sz):
	dt = transforms.Compose([
		transforms.Resize((sz, sz)),
		transforms.ToTensor(),
		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	])
	out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	out = dt(Image.fromarray(out)).float().unsqueeze(0)
	return out

def generate(netG, x, z, oc, sz, device):
	out = netG(x.to(device), z.to(device)).cpu().detach().numpy()
	out = out.reshape(oc, sz, sz).transpose(1, 2, 0)
	out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
	out = (out + 1) / 2.0
	return out

def interpolation(start, end, step_num, cur_step):
	return start * ((step_num - cur_step) / step_num) + end * (cur_step / step_num)

cv2.namedWindow('Input')
cv2.namedWindow('Output')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nz = 8
input_img_dir = 'samples'
input_img_list = os.listdir(input_img_dir)
model_path = 'saved/.pth'

sz, ic, oc, norm_type = 256, 3, 3, 'instancenorm'
netG = UNet_G(ic, oc, sz, nz, norm_type).to(device)
netG.load_state_dict(torch.load(model_path, map_location = 'cpu'))
netG.eval()

cnt, total_num = 0, 10
image = get_image(input_img_dir, input_img_list, cnt, sz)
noise = generate_noise(1, nz, device)
img = transform_image(image, sz)
out = generate(netG, img, noise, oc, sz, device)

while(1):
	cv2.imshow('Input', image)
	cv2.imshow('Output', out)

	key = cv2.waitKey(1) & 0xFF

	if(key == ord('q')):
		break

	elif(key == ord('r')):
		noise = generate_noise(1, nz, device)
		out = generate(netG, img, noise, oc, sz, device)

	elif(key == ord('t')):
		en = generate_noise(1, nz, device)
		sn = copy.deepcopy(noise)
		for i in range(10):
			cur_noise = interpolation(sn, en, 10, i+1)
			out = generate(netG, img, cur_noise, oc, sz, device)
			cv2.imshow('Input', image)
			cv2.imshow('Output', out)
			cv2.waitKey(1)
		noise = copy.deepcopy(en)

	elif(key == ord('e')):
		cnt += 1
		if(cnt>=total_num):
			cnt = 0
		image = get_image(input_img_dir, input_img_list, cnt, sz)
		img = transform_image(image, sz)
		out = generate(netG, img, noise, oc, sz, device)

cv2.destroyAllWindows()