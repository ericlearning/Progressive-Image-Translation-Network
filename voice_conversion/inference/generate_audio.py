import warnings
import librosa
import torch
import torch.nn as nn
from torchvision import transforms
import os, cv2, sys
import numpy as np
sys.path.append('..')
from architectures.baseline.unet import UNet_G
from architectures.baseline.resnet import ResNet_G
from utils.griffin_lim import *
from utils.inference_utils import *
from utils.utils import generate_noise
from PIL import Image
from tqdm import tqdm

# disable warnings caused by scipy
warnings.filterwarnings('ignore')

def spec_from_path_to_path(path1, path2):
	y = read_audio(path1, sample_rate, pre_emphasis_rate)
	mel = get_mel(get_stft(y, n_fft, win_length, hop_length), sample_rate, n_fft, n_mels, power, shrink_size)
	spec = mel_to_spectrogram(mel, threshold, path2)
	return spec

def generate_all(cnt):
	# source : loads audio, saves as spec
	path1 = os.path.join(source_wav_dir, source_wav_list[cnt])
	path2 = os.path.join(source_spec_save_dir, source_wav_list[cnt][:-4]+'.png')
	_ = spec_from_path_to_path(path1, path2)
	spec_src, ratio = get_image(path2, sz, resize_input)
	spec_src = transform_image(spec_src, sz, ic, resize_input)

	# source : loads spec, saves as audio
	path3 = os.path.join(source_wav_save_dir, source_wav_list[cnt])
	spec = cv2.imread(path2)
	stft = mel_to_stft(spectrogram_img_to_mel(spec, threshold), sample_rate, n_fft, n_mels, shrink_size, power)
	wave = griffin_lim(stft, griffin_lim_iter, n_fft, win_length, hop_length, pre_emphasis_rate)
	librosa.output.write_wav(path3, wave, sample_rate, norm = True)

	# target : loads audio, saves as spec
	path1 = os.path.join(target_wav_dir, source_wav_list[cnt])
	path2 = os.path.join(target_spec_save_dir, source_wav_list[cnt][:-4]+'.png')
	_ = spec_from_path_to_path(path1, path2)

	# target : loads spec, saves as audio
	path3 = os.path.join(target_wav_save_dir, source_wav_list[cnt])
	spec = cv2.imread(path2)
	stft = mel_to_stft(spectrogram_img_to_mel(spec, threshold), sample_rate, n_fft, n_mels, shrink_size, power)
	wave = griffin_lim(stft, griffin_lim_iter, n_fft, win_length, hop_length, pre_emphasis_rate)
	librosa.output.write_wav(path3, wave, sample_rate, norm = True)

	for i in range(noise_per_image):
		# path to save generated spec
		path3 = os.path.join(out_spec_save_dir, source_wav_list[cnt][:-4] + '-' + str(i) + '.png')
		noise = generate_noise(1, nz, device)
		# generate spec
		out = generate(netG, spec_src, noise, oc, sz, device)
		# save it in the path
		cv2.imwrite(path3, out)

		# read the generated spec
		spec = cv2.imread(path3)
		# changes the size of the generated spec to the size of the spec_src (input)
		spec = cv2.resize(spec, (0, 0), fx = 1/ratio, fy = 1)
		# makes it stft, then wave
		stft = mel_to_stft(spectrogram_img_to_mel(spec, threshold), sample_rate, n_fft, n_mels, shrink_size, power)
		wave = griffin_lim(stft, griffin_lim_iter, n_fft, win_length, hop_length, pre_emphasis_rate)
		# saves the wave
		path4 = os.path.join(out_wav_save_dir, source_wav_list[cnt][:-4] + '-' + str(i) + '.wav')
		librosa.output.write_wav(path4, wave, sample_rate, norm = True)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nz, noise_per_image = 8, 20

source_wav_dir = 'samples/source'
source_wav_list = os.listdir(source_wav_dir)
target_wav_dir = 'samples/target'

source_spec_save_dir = 'generated/spec/source'
target_spec_save_dir = 'generated/spec/target'
out_spec_save_dir = 'generated/spec/output'


source_wav_save_dir = 'generated/wav/source'
target_wav_save_dir = 'generated/wav/target'
out_wav_save_dir = 'generated/wav/output'


model_path = 'saved/.pth'

resize_input = True

sample_rate = 22050
pre_emphasis_rate = 0.97
n_fft = 2048
win_length = 1000
hop_length = 250
n_mels = 256
power = 1
shrink_size = 1
threshold = 5
griffin_lim_iter = 100

sz, ic, oc, use_bn, norm_type = 256, 1, 1, True, 'instancenorm'
netG = UNet_G(ic, oc, sz, nz, use_bn, norm_type).to(device)
# netG = ResNet_G(ic, oc, sz, nz = nz, norm_type = norm_type).to(device)
netG.load_state_dict(torch.load(model_path, map_location = 'cpu'))
netG.eval()

for cnt in tqdm(range(len(source_wav_list))):
	generate_all(cnt)

