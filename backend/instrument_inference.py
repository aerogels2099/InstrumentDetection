import torch
import librosa
import numpy as np
from instrument_cnn import CRNN
from instrument_labels import LABELS

INSTRUMENTS = LABELS

SR = 22050
DURATION = 5
N_MELS = 128
HOP_DURATION = 2.5
THRESHOLD = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_mel(y, sr=SR, duration=DURATION, n_mels=N_MELS):
    max_len = int(sr * duration)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel)

    delta = librosa.feature.delta(mel_db)
    delta2 = librosa.feature.delta(mel_db, order=2)

    mel_stack = np.stack([mel_db, delta, delta2])
    mel_stack = (mel_stack - mel_stack.mean()) / (mel_stack.std() + 1e-6)

    mel_tensor = torch.tensor(mel_stack, dtype=torch.float32).unsqueeze(0).to(device)
    return mel_tensor

def load_model(model_path="instrument_crnn.pth", device=device):
    num_classes = len(INSTRUMENTS)
    model = CRNN(num_classes=num_classes, n_mels=N_MELS)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_instruments(file_path, model, window_duration=DURATION, hop_duration=HOP_DURATION, threshold=THRESHOLD):
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    window_samples = int(window_duration * sr)
    hop_samples = int(hop_duration * sr)
    n_windows = max(1, (len(y) - window_samples) // hop_samples + 1)

    probs_accum = np.zeros(len(INSTRUMENTS))

    for i in range(n_windows):
        start = i * hop_samples
        end = start + window_samples
        window = y[start:end]
        mel = extract_mel(window)
        with torch.no_grad():
            logits = model(mel)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        probs_accum += probs

    probs_accum /= n_windows

    detected = [INSTRUMENTS[i] for i, p in enumerate(probs_accum) if p >= threshold]

    return detected, probs_accum