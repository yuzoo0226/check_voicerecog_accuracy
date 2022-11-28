import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def wav_read(path):
    wave, fs = sf.read(path) #音データと周波数を読み込む
    return wave, fs

if __name__ == "__main__":
    path = "../test_data/aiden_nr.wav"
    wave, fs = wav_read(path)
    time = np.arange(0,len(wave))/fs
    plt.plot(time, wave)
    plt.show()
    plt.close()