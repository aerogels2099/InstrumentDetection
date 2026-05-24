import torch
import librosa
import numpy as np
from backend.model import InstrumentCNN14
from backend.instrument_labels import LABELS

INSTRUMENTS = LABELS

SR = 32000
DURATION = 5.0
N_MELS = 64
HOP_DURATION = 2.5
THRESHOLD = 0.5


def extract_mel(y, sr=SR, duration=DURATION, n_mels=N_MELS, device="cpu"):
    max_len = int(sr * duration)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=320,
        fmin=50,
        fmax=14000,
    )
    mel_db = librosa.power_to_db(mel, ref=max(mel.max(), 1e-10))

    # (1, time_frames, mel_bins) — PANNs format
    x = mel_db.T[np.newaxis, :, :]
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)


def load_model(model_path, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    num_classes = checkpoint.get("num_classes", len(INSTRUMENTS))

    model = InstrumentCNN14(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    thresholds = checkpoint.get("thresholds", None)
    return model, thresholds


def predict_instruments(
    file_path, model, thresholds=None, device="cpu",
    window_duration=DURATION, hop_duration=HOP_DURATION, threshold=THRESHOLD,
):
    y, sr = librosa.load(file_path, sr=SR, mono=True)

    window_samples = int(window_duration * sr)
    hop_samples = int(hop_duration * sr)

    if len(y) <= window_samples:
        windows_to_process = [y]
    else:
        n_windows = max(1, (len(y) - window_samples) // hop_samples + 1)
        windows_to_process = []
        for i in range(n_windows):
            start = i * hop_samples
            end = start + window_samples
            if end <= len(y):
                windows_to_process.append(y[start:end])
            else:
                window = y[start:]
                window = np.pad(window, (0, window_samples - len(window)))
                windows_to_process.append(window)

    probs_accum = np.zeros(len(INSTRUMENTS))

    for window in windows_to_process:
        mel = extract_mel(window, device=device)
        with torch.no_grad():
            logits = model(mel)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        probs_accum += probs

    probs_accum /= len(windows_to_process)

    if thresholds is not None:
        detected = [
            INSTRUMENTS[i]
            for i, (p, t) in enumerate(zip(probs_accum, thresholds))
            if p >= t
        ]
    else:
        detected = [
            INSTRUMENTS[i]
            for i, p in enumerate(probs_accum)
            if p >= threshold
        ]

    return detected, probs_accum
