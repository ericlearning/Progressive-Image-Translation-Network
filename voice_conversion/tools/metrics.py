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

def calculate_mfcc(y, sample_rate, n_mfcc, ignore_first = True):
	m = librosa.feature.mfcc(y = y, sr = sample_rate, n_mfcc = n_mfcc)
	if(ignore_first):
		return m[:, 1:]
	return m

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