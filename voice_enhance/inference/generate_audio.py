import warnings
import torch
import torch.nn as nn
import torchaudio
from torchvision import transforms
import os, cv2, sys
import numpy as np
sys.path.append('..')
from architectures.baseline.unet import UNet_G
from utils.griffin_lim import read_audio
from tqdm import tqdm

# disable warnings caused by scipy
warnings.filterwarnings('ignore')

def generate_all(cnt):
	# source : loads audio, saves as spec
	path_src = os.path.join(source_wav_dir, source_wav_list[cnt])
	path_out = os.path.join(output_wav_dir, source_wav_list[cnt])

	x, sr = torchaudio.load(path_src)
	if(sr != sample_rate):
		print('Warning: The sample rates are not equal.')
	
	x = x.float().unsqueeze(0).to(device)
	y = netG(x).reshape(1, -1)
	torchaudio.save_encinfo(path_out, y)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
source_wav_dir = 'samples/source'
source_wav_list = os.listdir(source_wav_dir)
output_wav_dir = 'samples/output'
model_path = 'saved/.pth'

sample_rate = 22050
ic, oc, use_bn, norm_type = 1, 1, True, 'instancenorm'
netG = UNet_G(ic, oc, use_bn, use_sn = False, norm_type = norm_type).to(device)
netG.load_state_dict(torch.load(model_path, map_location = 'cpu'))
netG.eval()

for cnt in tqdm(range(len(source_wav_list))):
	generate_all(cnt)

