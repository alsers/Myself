import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pydub import AudioSegment

# Calculate and plot spectrogram for a WAV audio file
def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200  # length of each window segment
    fs = 8000
    noverlap = 120  # Overlap between windows
    nchannels = data.ndim  # same as columns_num
    if nchannels == 1: ## ??? 2021/8/11/0:59 ???
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap=noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:, 0], nfft, fs, noverlap=noverlap)        
    plt.show()
    return pxx



# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    # audio sampled at 44100Hz
    return rate, data


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def load_raw_audio(path):
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir(path + "activates"):
       if filename.endswith("wav"):  # endSwith !!
            activate = AudioSegment.from_wav(path + 'activates/' + filename)
            activates.append(activate)
    for filename in os.listdir(path + "backgrounds"):
        if filename.endswith("wav") :
            background = AudioSegment.from_wav(path + 'backgrounds/' + filename)
            backgrounds.append(background)
    for filename in os.listdir(path + "negatives"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav(path + 'negatives/' + filename)
            negatives.append(negative)
    return activates, negatives, backgrounds