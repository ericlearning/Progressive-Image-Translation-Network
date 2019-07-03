import cv2
import librosa
import numpy as np
import copy
import scipy
from tqdm import tqdm

# parameters
audio_path = 'input_wav/input.wav'
rec_audio_path1 = 'rec_wav/reconstruction.wav'
rec_audio_path2 = 'mel_rec_wav/reconstruction.wav'
sample_rate = 22050
pre_emphasis_rate = 0.97
n_fft = 2048
win_length = 1000
hop_length = 250
n_mels = 256
power = 1
shrink_size = 4

griffin_lim_iter = 300

threshold = 5

# options : save as spectrogram
get_stft_spec = True
stft_spec_name = 'spec/input_stft.png'

# options : load the spectrogram and use as stft
get_spec_stft = True
spec_stft_name = 'spec/input_stft.png'

# options : save the mel spectrogram
get_mel_spec = True
mel_spec_name = 'mel_spec/input_mel.png'

# options : load the mel spectrogram and use as stft
get_mel_spec_stft = True
mel_spec_stft_name = 'mel_spec/input_mel.png'

# get the time_series data of raw wav audio
y, _ = librosa.load(path = audio_path, sr = sample_rate, mono = True)

# pre-emphasis increases the amplitude of high frequency bands
# while decreasing the amplitude of low frequency bands
# y_t = x_t - a * x_t-1
y = np.append(y[0], y[1:] - pre_emphasis_rate * y[:-1])

# now, get the spectrogram
# spectrogram(signal, window) = abs(STFT(signal, window))
# hop length : samples between frames, usually n_fft / 4
# (1 + n_fft / 2, t)
stft = np.abs(librosa.core.stft(y, n_fft = n_fft, win_length = win_length, hop_length = hop_length))
print(stft.shape)

# converts a filterbank for mel conversion
# (n_mels, 1 + n_fft / 2)
mel_bank = librosa.filters.mel(sr = sample_rate, n_fft = n_fft, n_mels = n_mels)

# converts fft to mel using the filterbank
# (n_mels, 1 + n_fft / 2) x (1 + n_fft / 2, t) = (n_mels, t)
mel = np.dot(mel_bank, stft**power).astype('float32')

# shrink the x-axis size of mel by the shrink_size
mel = scipy.ndimage.zoom(mel, zoom = [1.0, 1 / shrink_size])
mel[mel < 1e-8] = 1e-8

# optional : get spectrogram
if(get_stft_spec):
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
	cv2.imwrite(stft_spec_name, save_img)

# optional : get the stft back from the spectrogram
if(get_spec_stft):
	# convert back to [0, -threshold]
	spec_img = cv2.cvtColor(cv2.imread(spec_stft_name), cv2.COLOR_BGR2GRAY)
	print(np.max(spec_img), np.min(spec_img))
	spec_img = (spec_img.astype('float32')) / 255.0 * (-threshold)
	print(np.max(spec_img), np.min(spec_img))

	# power 10
	stft_spec = np.power(10, spec_img)
	#print(np.max(spec_img), np.min(spec_img))

	# //TODO
	stft = stft_spec
	print(stft.shape)

# optional : get mel spectrogram
if(get_mel_spec):
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
	cv2.imwrite(mel_spec_name, save_img)

# optional : get the stft back from the mel spectrogram
if(get_mel_spec_stft):
	# convert back to [0, -threshold]
	spec_img = cv2.cvtColor(cv2.imread(mel_spec_stft_name), cv2.COLOR_BGR2GRAY)
	print(np.max(spec_img), np.min(spec_img))
	spec_img = (spec_img.astype('float32')) / 255.0 * (-threshold)
	print(np.max(spec_img), np.min(spec_img))

	# power 10
	mel_spec = np.power(10, spec_img)
	#print(np.max(spec_img), np.min(spec_img))

	# //TODO
	mel = mel_spec
	print(mel.shape)

# increase the x-axis size of mel by the shrink_size
mel_to_stft = scipy.ndimage.zoom(mel.astype('float32'), zoom = [1.0, shrink_size])
mel_to_stft = np.dot(mel_bank.T, mel_to_stft)**(1.0/power)

# griffin-lim algorithm : stft
tmp = copy.deepcopy(stft)
for _ in tqdm(range(griffin_lim_iter)):
	tmp1 = librosa.core.istft(tmp, win_length = win_length, hop_length = hop_length)
	tmp2 = librosa.core.stft(tmp1, n_fft = n_fft, win_length = win_length, hop_length = hop_length)
	tmp3 = tmp2 / (np.maximum(1e-8, np.abs(tmp2)))
	tmp = stft * tmp3

wave1 = np.real(librosa.core.istft(tmp, win_length = win_length, hop_length = hop_length))

print(np.max(stft), np.min(stft))
print('-'*19)
print(np.max(mel_to_stft), np.min(mel_to_stft))

# griffin-lim algorithm : mel -> stft
tmp = copy.deepcopy(mel_to_stft)
for _ in tqdm(range(griffin_lim_iter)):
	tmp1 = librosa.core.istft(tmp, win_length = win_length, hop_length = hop_length)
	tmp2 = librosa.core.stft(tmp1, n_fft = n_fft, win_length = win_length, hop_length = hop_length)
	tmp3 = tmp2 / (np.maximum(1e-8, np.abs(tmp2)))
	tmp = mel_to_stft * tmp3

wave2 = np.real(librosa.core.istft(tmp, win_length = win_length, hop_length = hop_length))


librosa.output.write_wav(rec_audio_path1, wave1, sample_rate, norm = True)
librosa.output.write_wav(rec_audio_path2, wave2, sample_rate, norm = True)

