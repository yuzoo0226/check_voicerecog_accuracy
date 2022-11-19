import glob
from argparse import ArgumentParser
from os.path import join
import time

import torch
from soundfile import write
from torchaudio import load
from tqdm import tqdm

import sys
sys.path.append("./references/sgmse")
from sgmse.model import ScoreModel
from sgmse.util.other import ensure_dir, pad_spec

GP_CHECKPOINT_FILE = "./ckpt/train_wsj0_2cta4cov_epoch=159.ckpt"
GP_CORRECTOR = "ald"
GP_N = 30
GP_SNR = 0.5
GP_CORRECT_STEPS = 1

class SpeechEnhancement():
    def __init__(self) -> None:
        # Load score model
        self.checkpoint_file = GP_CHECKPOINT_FILE
        self.correct_cls = GP_CORRECTOR
        self.N = GP_N
        self.snr = GP_SNR
        self.correct_steps = GP_CORRECT_STEPS

        self.model_sgmse = ScoreModel.load_from_checkpoint(self.checkpoint_file, base_dir='', batch_size=16, num_workers=0, kwargs=dict(gpu=False))
        self.model_sgmse.eval(no_ema=False)
        self.model_sgmse.cuda()

    def speech_enhancement_sgmse(self, noisy_file) -> None:
        start = time.time()
        filename = noisy_file.split('/')[-1]
        
        # Load wav
        y, _ = load(noisy_file) 
        T_orig = y.size(1)   

        # Normalize
        norm_factor = y.abs().max()
        y = y / norm_factor
        
        # Prepare DNN input
        Y = torch.unsqueeze(self.model_sgmse._forward_transform(self.model_sgmse._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        
        # Reverse sampling
        sampler = self.model_sgmse.get_pc_sampler(
            'reverse_diffusion', self.correct_cls, Y.cuda(), N=self.N, 
            corrector_steps=self.correct_steps, snr=self.snr)
        sample, _ = sampler()
        
        # Backward transform in time domain
        x_hat = self.model_sgmse.to_audio(sample.squeeze(), T_orig)

        # Renormalize
        x_hat = x_hat * norm_factor
        end = time.time()
        print(end - start)

        # Write enhanced wav file
        write("./wave_data/BITEC_wavedata/BITEC_name_car_near/wo_nr/aiden_se.wav", x_hat.cpu().numpy(), 16000)


if __name__ == "__main__":
    se_sgmse = SpeechEnhancement()
    se_sgmse.speech_enhancement_sgmse("./wave_data/BITEC_wavedata/BITEC_name_car_near/wo_nr/aiden.wav")