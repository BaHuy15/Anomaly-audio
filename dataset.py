import torch
from torch.utils.data import Dataset
import librosa
import re
import numpy as np
import glob
import torch.nn as nn


class args:
    frames=5
    n_mels=64
    frames=5
    n_fft=1024
    hop_length=512
    file_path=glob.glob('/home/tonyhuy/my_project/audio_classification/content/DATASET_FINAL/OK/train/*.wav')

class ASDDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.file_list = self.args.file_path


    def extract_signal_features(self,signal, sr, n_mels=64, frames=5, n_fft=1024, hop_length=512):
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
    
    def scale_minmax(self,X, min=0.0, max=1.0):
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
    def load_sound_file(self,wav_name, mono=False, channel=0):
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
            sampling_rate (float) - sampling rate detected in the file
    """
        multi_channel_data, sampling_rate = librosa.load(wav_name, sr=None, mono=mono)
        signal = np.array(multi_channel_data)#[channel, :]

        return signal, sampling_rate


    def __getitem__(self, item):
        data_item = self.file_list[item]
        #Load file 
        signal,sr=self.load_sound_file(data_item)

        # Extract features
        features=self.extract_signal_features(signal,sr,self.args.n_mels, self.args.frames, self.args.n_fft, self.args.hop_length)

        #Normalize features
        features = self.scale_minmax(features, 0, 1).astype(np.float)#.astype(np.uint8)
        # Convert to torch
        features= torch.from_numpy(features).float()

        return features

    def __len__(self):
        return len(self.file_list)

    

def main(args):
    train_set = ASDDataset(args)
    train_ = torch.utils.data.DataLoader(
                train_set,
                batch_size=128,
                shuffle=True,
                num_workers=20,
                pin_memory=True,
                drop_last=True
        )
    x=train_set.__getitem__(3)
    print(x.shape,type(x))
    model= AE()
    out=model(x)
    print(out.shape)

if __name__ == '__main__':
    main(args)
