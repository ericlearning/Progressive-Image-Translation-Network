import copy
import torch
import torch.nn as nn
from torchvision import transforms
import os, cv2, sys
import numpy as np
from utils import *
sys.path.append('..')
from architectures.architecture import UNet_G
from utils.griffin_lim import *
from utils.utils import generate_noise
from PIL import Image

cv2.namedWindow('Input')
cv2.namedWindow('Output')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nz = 8
input_img_dir = 'samples'
input_img_list = os.listdir(input_img_dir)
model_path = 'saved/.pth'

sz, ic, oc, use_bn, norm_type = 256, 3, 3, True, 'instancenorm'
netG = UNet_G(ic, oc, sz, nz, use_bn, norm_type).to(device)
netG.load_state_dict(torch.load(model_path, map_location = 'cpu'))
netG.eval()

cnt, total_num = 0, 10
image, _ = get_image(os.path.join(input_img_dir, input_img_list[cnt]), sz)
image = transform_image(image, sz, ic)
noise = generate_noise(1, nz, device)
out = generate(netG, image, noise, oc, sz, device)

while(1):
	cv2.imshow('Input', image)
	cv2.imshow('Output', out)

	key = cv2.waitKey(1) & 0xFF

	if(key == ord('q')):
		break

	elif(key == ord('r')):
		noise = generate_noise(1, nz, device)
		out = generate(netG, image, noise, oc, sz, device)

	elif(key == ord('t')):
		en = generate_noise(1, nz, device)
		sn = copy.deepcopy(noise)
		for i in range(10):
			cur_noise = interpolation(sn, en, 10, i+1)
			out = generate(netG, image, cur_noise, oc, sz, device)
			cv2.imshow('Input', image)
			cv2.imshow('Output', out)
			cv2.waitKey(1)
		noise = copy.deepcopy(en)

	elif(key == ord('e')):
		cnt += 1
		if(cnt>=total_num):
			cnt = 0
		image, _ = get_image(os.path.join(input_img_dir, input_img_list[cnt]), sz)
		image = transform_image(image, sz, ic)
		out = generate(netG, image, noise, oc, sz, device)

cv2.destroyAllWindows()