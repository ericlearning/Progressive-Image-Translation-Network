import os
import torch
import random
import librosa
import numpy as np
import torchaudio
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

class Dataset():
	def __init__(self, train_dir, basic_types = None, shuffle = True):
		self.train_dir = train_dir
		self.basic_types = basic_types
		self.shuffle = shuffle

	def get_loader(self, sz, bs, num_workers = 1):
		train_dataset = Audio_Dataset(self.train_dir[0], self.train_dir[1])
		train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = self.shuffle, num_workers = num_workers)

		returns = (train_loader)

		return returns

class Audio_Dataset():
	def __init__(self, input_dir, target_dir):
		self.input_dir = input_dir
		self.target_dir = target_dir

		self.audio_name_list = []
		for file in os.listdir(input_dir):
			if(file.endswith('.wav')):
				self.audio_name_list.append(file)

	def __len__(self):
		return len(self.audio_name_list)

	def __getitem__(self, idx):
		input_wav, sr1 = torchaudio.load(os.path.join(self.input_dir, self.audio_name_list[idx]))
		target_wav, sr2 = torchaudio.load(os.path.join(self.target_dir, self.audio_name_list[idx]))

		# size : (1, n)
		wave_shape_1 = input_wav.shape[1]
		wave_shape_2 = target_wav.shape[1]

		if(sr1 != sr2):
			print('Warning: SampleRate does not match')
			print(sr1, sr2)

		if(wave_shape_1 != wave_shape_2):
			print('Warning: Wave Shape does not match')
			print(wave_shape_1, wave_shape_2)

		t1 = random.randint(0, wave_shape+1-16384)
		t2 = t1+16384
		if(t2>wave_shape):
			t2 = wave_shape

		input_wav = input_wav[:, t1:t2]
		target_wav = target_wav[:, t1:t2]

		if(t2 == wave_shape):
			#input_wav = librosa.util.fix_length(input_wav, 16384)
			#target_wav = librosa.util.fix_length(target_wav, 16384)
			input_wav = torchaudio.functional.pad_trim(input_wav, 0, 16384, 1, 0)
			target_wav = torchaudio.functional.pad_trim(target_wav, 0, 16384, 1, 0)

		sample = (input_wav, target_wav)
		return sample