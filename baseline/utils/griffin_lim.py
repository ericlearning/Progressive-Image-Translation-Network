import cv2
import librosa
import numpy as np
import copy
import scipy
from tqdm import tqdm

# sample_rate : 22050
# pre_emphasis_rate : 0.97
# n_fft : 2048
# win_length : 1000
# hop_length : 250
# n_mels : 256
# power : 1
# shrink_size : 4
# threshold : 5
# griffin_lim_iter : 100

def read_audio(audio_path, sample_rate, pre_emphasis_rate):
	# get the time_series data of raw wav audio
	y, _ = librosa.load(path = audio_path, sr = sample_rate, mono = True)

	# pre-emphasis increases the amplitude of high frequency bands
	# while decreasing the amplitude of low frequency bands
	# y_t = x_t - a * x_t-1
	y = np.append(y[0], y[1:] - pre_emphasis_rate * y[:-1])

	return y

def get_stft(y, n_fft, win_length, hop_length):
	# now, get the spectrogram
	# spectrogram(signal, window) = abs(STFT(signal, window))
	# hop length : samples between frames, usually n_fft / 4
	# (1 + n_fft / 2, t)
	stft = np.abs(librosa.core.stft(y, n_fft = n_fft, win_length = win_length, hop_length = hop_length))
	print(stft.shape)

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

def stft_to_spectrogram(stft, threshold, out_name):
	# transpose
	stft_spec = stft

	# normalize range
	print(np.max(stft_spec), np.min(stft_spec))
	stft_spec = stft_spec / np.max(stft_spec)
	print(np.max(stft_spec), np.min(stft_spec))

	# take log10
	stft_spec = np.log10(stft_spec)
	print(np.max(stft_spec), np.min(stft_spec))

	# apply threshold
	stft_spec[stft_spec < -threshold] = -threshold
	print(np.max(stft_spec), np.min(stft_spec))

	# convert to [0, 255] to save it as an image
	save_img = (stft_spec / (-threshold) * 255.0).astype('uint8')
	cv2.imwrite(out_name, save_img)

	return save_img

def spectrogram_to_stft(in_name, threshold):
	# convert back to [0, -threshold]
	spec_img = cv2.cvtColor(cv2.imread(in_name), cv2.COLOR_BGR2GRAY)
	print(np.max(spec_img), np.min(spec_img))
	spec_img = (spec_img.astype('float32')) / 255.0 * (-threshold)
	print(np.max(spec_img), np.min(spec_img))

	# power 10
	stft_spec = np.power(10, spec_img)
	#print(np.max(spec_img), np.min(spec_img))

	# //TODO
	stft = stft_spec
	print(stft.shape)

	return stft

def mel_to_spectrogram(mel, threshold, out_name):
	# transpose
	print(mel.shape, stft.shape)
	mel_spec = mel

	# normalize range
	print(np.max(mel_spec), np.min(mel_spec))
	mel_spec = mel_spec / np.max(mel_spec)
	print(np.max(mel_spec), np.min(mel_spec))

	# take log10
	mel_spec = np.log10(mel_spec)
	print(np.max(mel_spec), np.min(mel_spec))

	# apply threshold
	mel_spec[mel_spec < -threshold] = -threshold
	print(np.max(mel_spec), np.min(mel_spec))

	# convert to [0, 255] to save it as an image
	save_img = (mel_spec / (-threshold) * 255.0).astype('uint8')
	cv2.imwrite(out_name, save_img)

	return save_img

def spectrogram_to_mel(in_name, threshold):
	# convert back to [0, -threshold]
	spec_img = cv2.cvtColor(cv2.imread(in_name), cv2.COLOR_BGR2GRAY)
	print(np.max(spec_img), np.min(spec_img))
	spec_img = (spec_img.astype('float32')) / 255.0 * (-threshold)
	print(np.max(spec_img), np.min(spec_img))

	# power 10
	mel_spec = np.power(10, spec_img)
	#print(np.max(spec_img), np.min(spec_img))

	# //TODO
	mel = mel_spec
	print(mel.shape)

	return mel

def spectrogram_img_to_mel(spectrogram, threshold):
	# convert back to [0, -threshold]
	spec_img = cv2.cvtColor(spectrogram, cv2.COLOR_BGR2GRAY)
	print(np.max(spec_img), np.min(spec_img))
	spec_img = (spec_img.astype('float32')) / 255.0 * (-threshold)
	print(np.max(spec_img), np.min(spec_img))

	# power 10
	mel_spec = np.power(10, spec_img)
	#print(np.max(spec_img), np.min(spec_img))

	# //TODO
	mel = mel_spec
	print(mel.shape)

	return mel

def mel_to_stft(mel, sample_rate, n_fft, n_mels, shrink_size, power):
	mel_bank = librosa.filters.mel(sr = sample_rate, n_fft = n_fft, n_mels = n_mels)

	# increase the x-axis size of mel by the shrink_size
	mel_to_stft = scipy.ndimage.zoom(mel.astype('float32'), zoom = [1.0, shrink_size])
	mel_to_stft = np.dot(mel_bank.T, mel_to_stft)**(1.0/power)

	return mel_to_stft

def griffin_lim(input_, griffin_lim_iter, n_fft, win_length, hop_length):
	tmp = copy.deepcopy(input_)
	for _ in tqdm(range(griffin_lim_iter)):
		tmp1 = librosa.core.istft(tmp, win_length = win_length, hop_length = hop_length)
		tmp2 = librosa.core.stft(tmp1, n_fft = n_fft, win_length = win_length, hop_length = hop_length)
		tmp3 = tmp2 / (np.maximum(1e-8, np.abs(tmp2)))
		tmp = input_ * tmp3

	wave = np.real(librosa.core.istft(tmp, win_length = win_length, hop_length = hop_length))

	return wave
