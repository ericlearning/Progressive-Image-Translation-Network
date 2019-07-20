import os
import torch
import random
import librosa
import numpy as np
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
		input_wav = librosa.core.load(os.path.join(self.input_dir, self.audio_name_list[idx]), sr = 22050, mono = True)
		target_wav = librosa.core.load(os.path.join(self.target_dir, self.audio_name_list[idx]), sr = 22050, mono = True)

		# size : (n, )
		wave_shape = input_wav.shape[0]
		t1 = random.randint(0, wave_shape+1-16384)
		t2 = t1+16384
		if(t2>wave_shape):
			t2 = wave_shape

		input_wav = input_wav[t1:t2]
		target_wav = target_wav[t1:t2]

		if(t2 == wave_shape):
			input_wav = librosa.util.fix_length(input_wav, 16384)
			target_wav = librosa.util.fix_length(target_wav, 16384)



		sample = (input_wav, target_wav)
		return sample