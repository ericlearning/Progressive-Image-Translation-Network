import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

def get_image(img_name, sz, resize_input):
	image = cv2.imread(img_name)
	if(resize_input):
		ratio = sz / image.shape[1]
		image = cv2.resize(image, (sz, sz))
	else:
		ratio = 1
	return image, ratio

def transform_image(image, sz, ic, resize_input):
	if(ic == 1):
		dt = transforms.Compose([
			transforms.Grayscale(1),
			transforms.ToTensor(),
			transforms.Normalize([0.5], [0.5])
		])
	else:
		dt = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])

	out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	resizer = transforms.Resize((sz, sz))
	if(resize_input):
		out = resizer(out)	
	out = dt(Image.fromarray(out)).float().unsqueeze(0)
	return out

def interpolation(start, end, step_num, cur_step):
	return start * ((step_num - cur_step) / step_num) + end * (cur_step / step_num)

def generate(netG, x, z, oc, device):
	out = netG(x.to(device), z).cpu().detach().numpy()
	out = out.reshape(oc, out.shape[2], out.shape[3]).transpose(1, 2, 0)
	if(oc > 1):
		out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
	out = ((out + 1) / 2.0 * 255.0).astype('uint8')
	return out