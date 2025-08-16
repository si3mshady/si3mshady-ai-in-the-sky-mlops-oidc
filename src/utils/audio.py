import numpy as np, librosa

def wav_to_melspec(y, sr, n_fft=1024, hop_length=256, n_mels=64):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
    Sdb = librosa.power_to_db(S, ref=np.max)
    return Sdb.astype("float32")

def normalize_spec(S):
    m, s = S.mean(), S.std() + 1e-6
    return (S - m) / s

