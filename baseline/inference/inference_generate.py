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
	out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
	out = (out + 1) / 2.0
	return out

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nz, noise_per_image = 8, 20
input_img_dir = 'samples'
output_img_dir = 'generated'
input_img_list = os.listdir(input_img_dir)
model_path = 'saved/.pth'

sz, ic, oc, norm_type = 256, 3, 3, 'instancenorm'
netG = UNet_G(ic, oc, sz, nz, norm_type).to(device)
netG.load_state_dict(torch.load(model_path, map_location = 'cpu'))
netG.eval()

out_cnt = 0
for cnt in range(len(input_img_list)):
	image = get_image(input_img_dir, input_img_list, cnt, sz)
	image = transform_image(image, sz)
	for i in range(noise_per_image):
		noise = generate_noise(1, nz, device)
		out = generate(netG, image, noise, oc, sz, device)
		cv2.imwrite(os.path.join(output_img_dir, str(out_cnt)+'.png'), out)
