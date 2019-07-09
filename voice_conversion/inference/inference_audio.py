import copy
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

def spec_from_path(path):
	y = read_audio(path, sample_rate, pre_emphasis_rate)
	mel = get_mel(get_stft(y, n_fft, win_length, hop_length), sample_rate, n_fft, n_mels, power, shrink_size)
	spec = cv2.resize(cv2.cvtColor(mel_to_spectrogram(mel, threshold, None), cv2.COLOR_GRAY2RGB), (sz, sz))
	return spec

cv2.namedWindow('Source')
cv2.namedWindow('Target')
cv2.namedWindow('Output')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nz = 8
source_wav_dir = 'samples/source'
source_wav_list = os.listdir(source_wav_dir)
target_wav_dir = 'samples/target'

model_path = 'saved/.pth'

sample_rate = 22050
pre_emphasis_rate = 0.97
n_fft = 2048
win_length = 1000
hop_length = 250
n_mels = 256
power = 1
shrink_size = 3.5
threshold = 5
griffin_lim_iter = 100

sz, ic, oc, use_bn, norm_type = 256, 1, 1, True, 'instancenorm'
netG = UNet_G(ic, oc, sz, nz, use_bn, norm_type).to(device)
# netG = ResNet_G(ic, oc, sz, nz = nz, norm_type = norm_type).to(device)
netG.load_state_dict(torch.load(model_path, map_location = 'cpu'))
netG.eval()

cnt, total_num = 0, 10

noise = generate_noise(1, nz, device)

spec_src = spec_from_path(os.path.join(source_wav_dir, source_wav_list[cnt]))
spec_src_t = transform_image(spec_src, sz, ic)
spec_tar = spec_from_path(os.path.join(target_wav_dir, source_wav_list[cnt]))

out = generate(netG, spec_src_t, noise, oc, sz, device)

while(1):
	cv2.imshow('Source', spec_src)
	cv2.imshow('Target', spec_tar)
	cv2.imshow('Output', out)

	key = cv2.waitKey(1) & 0xFF

	if(key == ord('q')):
		break

	elif(key == ord('r')):
		noise = generate_noise(1, nz, device)
		out = generate(netG, spec_src_t, noise, oc, sz, device)

	elif(key == ord('t')):
		en = generate_noise(1, nz, device)
		sn = copy.deepcopy(noise)
		for i in range(10):
			cur_noise = interpolation(sn, en, 10, i+1)
			out = generate(netG, spec_src_t, cur_noise, oc, sz, device)
			cv2.imshow('Source', spec_src)
			cv2.imshow('Target', spec_tar)
			cv2.imshow('Output', out)
			cv2.waitKey(1)
		noise = copy.deepcopy(en)

	elif(key == ord('e')):
		cnt += 1
		if(cnt>=total_num):
			cnt = 0

		spec_src = spec_from_path(os.path.join(source_wav_dir, source_wav_list[cnt]))
		spec_src_t = transform_image(spec_src, sz, ic)
		spec_tar = spec_from_path(os.path.join(target_wav_dir, source_wav_list[cnt]))
		out = generate(netG, spec_src_t, noise, oc, sz, device)

cv2.destroyAllWindows()