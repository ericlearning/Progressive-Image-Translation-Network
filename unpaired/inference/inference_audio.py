import copy
import torch
import torch.nn as nn
from torchvision import transforms
import os, cv2, sys
import numpy as np
sys.path.append('..')
from architectures.architecture import UNet_G
from utils.griffin_lim import *
from utils.inference_utils import *
from utils.utils import generate_noise
from PIL import Image

cv2.namedWindow('Input')
cv2.namedWindow('Output')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nz = 8
input_wav_dir = 'samples'
input_wav_list = os.listdir(input_wav_dir)
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
netG.load_state_dict(torch.load(model_path, map_location = 'cpu'))
netG.eval()

cnt, total_num = 0, 10
y = read_audio(os.path.join(input_wav_dir, input_wav_list[cnt]), sample_rate, pre_emphasis_rate)
mel = get_mel(get_stft(y, n_fft, win_length, hop_length), sample_rate, n_fft, n_mels, power, shrink_size)
spec = cv2.resize(cv2.cvtColor(mel_to_spectrogram(mel, threshold, None), cv2.COLOR_GRAY2RGB), (sz, sz))
spec_t = transform_image(spec, sz, ic)
noise = generate_noise(1, nz, device)
out = generate(netG, spec_t, noise, oc, sz, device)

while(1):
	cv2.imshow('Input', spec)
	cv2.imshow('Output', out)

	key = cv2.waitKey(1) & 0xFF

	if(key == ord('q')):
		break

	elif(key == ord('r')):
		noise = generate_noise(1, nz, device)
		out = generate(netG, spec_t, noise, oc, sz, device)

	elif(key == ord('t')):
		en = generate_noise(1, nz, device)
		sn = copy.deepcopy(noise)
		for i in range(10):
			cur_noise = interpolation(sn, en, 10, i+1)
			out = generate(netG, spec_t, cur_noise, oc, sz, device)
			cv2.imshow('Input', image)
			cv2.imshow('Output', out)
			cv2.waitKey(1)
		noise = copy.deepcopy(en)

	elif(key == ord('e')):
		cnt += 1
		if(cnt>=total_num):
			cnt = 0
		y = read_audio(os.path.join(input_wav_dir, input_wav_list[cnt]), sample_rate, pre_emphasis_rate)
		mel = get_mel(get_stft(y, n_fft, win_length, hop_length), sample_rate, n_fft, n_mels, power, shrink_size)
		spec = cv2.resize(cv2.cvtColor(mel_to_spectrogram(mel, threshold, None), cv2.COLOR_GRAY2RGB), (sz, sz))
		spec_t = transform_image(spec, sz, ic)
		out = generate(netG, spec_t, noise, oc, sz, device)

cv2.destroyAllWindows()