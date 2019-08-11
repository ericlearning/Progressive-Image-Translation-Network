import os, cv2, sys
import time
import numpy as np
import sounddevice as sd
sys.path.append('..')
from architectures.baseline.resnet import ResNet_G
from utils.griffin_lim import *
from utils.inference_utils import *
from utils.utils import generate_noise

def record_audio():
	wav = sd.rec(63999, samplerate = 22050, channels=1, dtype = 'float64')
	print('Recording Initiated')
	sd.wait()
	print('Recording Complete')
	wav = wav.reshape(-1)
	return wav

def play_audio(wav, samplerate = 22050):
	print('Playing Audio')
	print(wav.shape)
	sd.play(wav, samplerate)
	sd.wait()
	print('Playing Audio Complete')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = 'model.state'

sz, nz, ic, oc, use_bn, norm_type = 256, None, 1, 1, True, 'instancenorm'
noise = generate_noise(1, nz, device)
netG = nn.DataParallel(ResNet_G(ic, oc, sz, nz = nz, norm_type = norm_type)).to(device)
netG.load_state_dict(torch.load(model_path, map_location = 'cpu')['netG_A2B'])
netG.eval()


y = record_audio()
y = librosa.util.normalize(y, norm = np.inf, axis = None)
print(np.max(y), np.min(y))
mel = get_mel(get_stft(y, 2048, 1000, 250), 22050, 2048, 256, 1, 1)
spec = mel_to_spectrogram(mel, 5, None)


print('Model Inference Start')
spec_t = transform_image(spec, 256, ic, resize_input = False)
out_spec = generate(netG, spec_t, noise, oc, device)
out_spec = out_spec.reshape(out_spec.shape[0], out_spec.shape[1])
print('Model Inference End')


print('Griffin Lim Process Start')
out_mel = spectrogram_img_to_mel(out_spec, 5, gray = True)
out_stft = mel_to_stft(out_mel, 22050, 2048, 256, 1, 1)
out = griffin_lim(out_stft, 300, 2048, 1000, 250, None)
out = librosa.util.normalize(out, norm = np.inf, axis = None)
print('Griffin Lim Process End')

out = out[500:]
play_audio(out.reshape(-1, 1), 22050)
print('-'*8, 'Process complete.')

cv2.namedWindow('Input')
cv2.namedWindow('Output')

while True:
	cv2.imshow('Input', spec)
	cv2.imshow('Output', out_spec)

	key = cv2.waitKey(1) & 0xFF
	if(key == ord('q')):
		break

cv2.destroyAllWindows()