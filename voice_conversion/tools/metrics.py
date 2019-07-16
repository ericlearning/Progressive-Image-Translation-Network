import cv2
import librosa
import numpy as np
import copy
import scipy
import math
from tqdm import tqdm

# sample_rate : 22050
# pre_emphasis_rate : 0.97
# n_fft : 2048
# win_length : 1000
# hop_length : 250
# n_mels : 256
# power : 1
# shrink_size : 3.5
# threshold : 5
# griffin_lim_iter : 100

def read_audio(audio_path, sample_rate, pre_emphasis_rate):
	# get the time_series data of raw wav audio
	y, _ = librosa.load(path = audio_path, sr = sample_rate, mono = True)
	return y

def get_stft(y, n_fft, win_length, hop_length):
	# now, get the spectrogram
	# spectrogram(signal, window) = abs(STFT(signal, window))
	# hop length : samples between frames, usually n_fft / 4
	# (1 + n_fft / 2, t)
	stft = np.abs(librosa.core.stft(y, n_fft = n_fft, win_length = win_length, hop_length = hop_length))

	return stft

def get_mel(stft, sample_rate, n_fft, n_mels, power, shrink_size):
	# converts a filterbank for mel conversion
	# (n_mels, 1 + n_fft / 2)
	mel_bank = librosa.filters.mel(sr = sample_rate, n_fft = n_fft, n_mels = n_mels)

	# converts stft to mel using the filterbank
	# (n_mels, 1 + n_fft / 2) x (1 + n_fft / 2, t) = (n_mels, t)
	mel = np.dot(mel_bank, stft**power).astype('float32')

	# shrink the x-axis size of mel by the shrink_size
	mel = scipy.ndimage.zoom(mel, zoom = [1.0, 1 / shrink_size])
	mel[mel < 1e-8] = 1e-8

	return mel

def calculate_mfcc(y, sample_rate, n_fft, n_mels, n_mffc, win_length, hop_length, power = 2)
	mel = librosa.power_to_db(get_mel(get_stft(y, n_fft, win_length, hop_length), sample_rate, n_fft, n_mels, power, 1))
	mfcc = np.dot(librosa.filters.dct(n_mffc, n_mels), mel)
	return mfcc

def calculate_mcd(s, t):
	if(s.shape[0] != t.shape[0]):
		return -1

	mcd_coefficient = 10.0 / math.log(10.0) * math.sqrt(2)
	mcd = 0
	for i in range(s.shape[0]):
		diff = s[i, :] - t[i, :]
		mcd += np.sqrt(np.inner(diff, diff))
	mcd = mcd * mcd_coefficient / (s.shape[0])
	return mcd

def wave_stretch(y1, y2):
	# y1 is the model output
	# y2 is the target
	# y1 will be stretched
	y1_len = y1.shape[0]
	y2_len = y2.shape[0]
	rate = y1_len / y2_len
	y1_stretched = librosa.effects.time_stretch(y1, rate)

	return y1_stretched, y2