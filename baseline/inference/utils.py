import cv2
import torch
import torch.nn as nn
from torchvision import transforms

def get_image(img_name, sz):
	image = cv2.imread(img_name)
	ratio = sz / image.shape[1]
	image = cv2.resize(image, (sz, sz))
	return image, ratio

def transform_image(image, sz, ic):
	if(ic == 1):
		dt = transforms.Compose([
			transforms.Resize((sz, sz)),
			transforms.Grayscale(1),
			transforms.ToTensor(),
			transforms.Normalize([0.5], [0.5])
		])
	else:
		dt = transforms.Compose([
			transforms.Resize((sz, sz)),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])
		
	out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	out = dt(Image.fromarray(out)).float().unsqueeze(0)
	return out

def interpolation(start, end, step_num, cur_step):
	return start * ((step_num - cur_step) / step_num) + end * (cur_step / step_num)

def generate(netG, x, z, oc, sz, device):
	out = netG(x.to(device), z).cpu().detach().numpy()
	out = out.reshape(oc, sz, sz).transpose(1, 2, 0)
	if(oc > 1):
		out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
	out = ((out + 1) / 2.0 * 255.0).astype('uint8')
	return out