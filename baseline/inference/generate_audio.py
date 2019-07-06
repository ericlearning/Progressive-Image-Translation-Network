import librosa
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nz, noise_per_image = 8, 20

input_wav_dir = 'samples/wav'
input_spec_save_dir = 'samples/spec'
output_wav_dir = 'generated/wav'
output_spec_save_dir = 'generated/spec'
model_path = 'saved/.pth'
input_wav_list = os.listdir(input_wav_dir)

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

out_cnt = 0
for cnt in range(len(input_wav_dir)):
	y = read_audio(os.path.join(input_wav_dir, input_wav_list[cnt]), sample_rate, pre_emphasis_rate)
	mel = get_mel(get_stft(y, n_fft, win_length, hop_length), sample_rate, n_fft, n_mels, power, shrink_size)
	spec = mel_to_spectrogram(mel, threshold, os.path.join(input_spec_save_dir, input_wav_list[cnt][:-4]+'.png'))

	image, ratio = get_image(os.path.join(input_spec_save_dir, input_wav_list[cnt][:-4]+'.png'), sz)
	image = transform_image(image, sz, ic)

	for i in range(noise_per_image):
		noise = generate_noise(1, nz, device)
		out = generate(netG, image, noise, oc, sz, device)
		cv2.imwrite(os.path.join(output_spec_save_dir, input_wav_list[cnt][:-4]+ '-' str(i) + '.png'), out)

		spec = cv2.imread(os.path.join(output_spec_save_dir, input_wav_list[cnt][:-4]+ '-' str(i) + '.png'))
		spec = cv2.resize(spec, (0, 0), fx = 1/ratio, fy = 1)
		stft = mel_to_stft(spectrogram_img_to_mel(spec, threshold), sample_rate, n_fft, n_mels, shrink_size, power)
		wave = griffin_lim(stft, griffin_lim_iter, n_fft, win_length, hop_length, pre_emphasis_rate)

		librosa.output.write_wav(os.path.join(output_wav_dir, input_wav_list[cnt][:-4]+ '-' str(i) + '.wav'), wave, sample_rate, norm = True)

