import librosa
import torch
import torch.nn as nn
from torchvision import transforms
import os, cv2, sys
import numpy as np
sys.path.append('..')
from architectures.unet import UNet_G
from architectures.resnet import ResNet_G
from utils.griffin_lim import *
from utils.inference_utils import *
from utils.utils import generate_noise
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nz, noise_per_image = 8, 20
input_img_dir = 'samples'
output_img_dir = 'generated'
input_img_list = os.listdir(input_img_dir)
model_path = 'saved/.pth'

sz, ic, oc, use_bn, norm_type = 256, 3, 3, True, 'instancenorm'
netG = UNet_G(ic, oc, sz, nz, use_bn, norm_type).to(device)
# netG = ResNet_G(ic, oc, sz, nz = nz, norm_type = norm_type).to(device)
netG.load_state_dict(torch.load(model_path, map_location = 'cpu'))
netG.eval()

out_cnt = 0
for cnt in range(len(input_img_list)):
	image, ratio = get_image(os.path.join(input_img_dir, input_img_list[cnt]), sz)
	image = transform_image(image, sz, ic)
	for i in range(noise_per_image):
		noise = generate_noise(1, nz, device)
		out = generate(netG, image, noise, oc, sz, device)
		cv2.imwrite(os.path.join(output_img_dir, str(out_cnt)+'.png'), out)
		out_cnt += 1