import torch
import glob
import torch.nn as nn
from model import AE
import numpy as np
import random
from dataset import ASDDataset
from collections import defaultdict
from datetime import timedelta
import time
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns

# https://www.kaggle.com/code/kimchanyoung/pytorch-anomaly-detection-with-autoencoder

class args:
    frames=5
    n_mels=64
    frames=5
    n_fft=1024
    hop_length=512
    lr=1e-02
    w_d = 1e-3 
    epochs =80
    seed=42
    batch_size=128
    file_path=glob.glob('/home/tonyhuy/my_project/audio_classification/content/DATASET_FINAL/NG/test/*.wav')

def extract_signal_features(signal, sr, n_mels=64, frames=5, n_fft=1024, hop_length=512):
    # https://github.com/aws-samples/sound-anomaly-detection-for-manufacturing/blob/main/tools/sound_tools.py
    """
    Extract features from a sound signal, given a sampling rate sr. This function 
    computes the Mel spectrogram in log scales (getting the power of the signal).
    Then we build N frames (where N = frames passed as an argument to this function):
    each frame is a sliding window in the temporal dimension.
    
    PARAMS
    ======
        signal (array of floats) - numpy array as returned by load_sound_file()
        sr (integer) - sampling rate of the signal
        n_mels (integer) - number of Mel buckets (default: 64)
        frames (integer) - number of sliding windows to use to slice the Mel spectrogram
        n_fft (integer) - length of the windowed signal to compute the short Fourier transform on
        hop_length (integer) - window increment when computing STFT
    """

    # Compute a mel-scaled spectrogram:
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to decibel (log scale for amplitude):
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Generate an array of vectors as features for the current signal:
    features_vector_size = log_mel_spectrogram.shape[1] - frames + 1
    
    # Skips short signals:
    dims = frames * n_mels
    if features_vector_size < 1:
        return np.empty((0, dims), np.float32)
    
    # Build N sliding windows (=frames) and concatenate them to build a feature vector:
    features = np.zeros((features_vector_size, dims), np.float32)
    for t in range(frames):
        features[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t:t + features_vector_size].T
        
    return features
    
def scale_minmax(X, min=0.0, max=1.0):
    """
    Minmax scaler for a numpy array
    
    PARAMS
    ======
        X (numpy array) - array to scale
        min (float) - minimum value of the scaling range (default: 0.0)
        max (float) - maximum value of the scaling range (default: 1.0)
    """
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled
def load_sound_file(wav_name, mono=False, channel=0):
    """
    Loads a sound file
    PARAMS
    ======
        wav_name (string) - location to the WAV file to open
        mono (boolean) - signal is in mono (if True) or Stereo (False, default)
        channel (integer) - which channel to load (default to 0)
    
    RETURNS
    =======
        signal (numpy array) - sound signal
        sampling_rate (float) - sampling rate detected in the file"""
    multi_channel_data, sampling_rate = librosa.load(wav_name, sr=None, mono=mono)
    signal = np.array(multi_channel_data)#[channel, :]

    return signal, sampling_rate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model=AE()
loaded_semantic = torch.load('/home/tonyhuy/my_project/anomaly_audio/checkpoint/epoch_79.pth', map_location='cpu')
model = model.to(device)
model.load_state_dict(loaded_semantic['state_dict'])
criterion = nn.MSELoss(reduction='mean')
loss_dist = []

for i in range(99):
    data_item=args.file_path[i]
    #Load file 
    signal,sr=load_sound_file(data_item)

    # Extract features
    features=extract_signal_features(signal,sr,args.n_mels, args.frames, args.n_fft, args.hop_length)

    #Normalize features
    features = scale_minmax(features, 0, 1).astype(np.float)#.astype(np.uint8)
    # Convert to torch
    features= torch.from_numpy(features).float()


    model.eval()
    sample = model(features.to(device))
    loss = criterion(features.to(device), sample)
    print('[LOSS] {}'.format(loss))
    loss_dist.append(loss.item())

lower_threshold = 0.0
upper_threshold = 0.17
plt.figure(figsize=(12,6))
plt.title('Loss Distribution')
sns.distplot(loss_dist,bins=100,kde=True, color='blue')
plt.axvline(upper_threshold, 0.0, 10, color='r')
plt.axvline(lower_threshold, 0.0, 10, color='b')
plt.savefig("/home/tonyhuy/my_project/anomaly_audio/result/squares.png") 
    
