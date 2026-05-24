import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset


class InstrumentDataset(Dataset):
    def __init__(self, csv_file, sr=32000, duration=5.0, n_mels=64, augment=False):
        self.data = pd.read_csv(csv_file)
        self.sr = sr
        self.duration = duration
        self.win_len = int(sr * duration)
        self.n_mels = n_mels
        self.augment = augment

        self.label_cols = [c for c in self.data.columns if c not in ("file_path", "duration")]

        self.samples = []
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            try:
                if "duration" in self.data.columns:
                    audio_duration = float(row["duration"])
                else:
                    audio_duration = librosa.get_duration(path=row["file_path"])

                if audio_duration <= duration:
                    self.samples.append((idx, 0.0, audio_duration))
                else:
                    if augment:
                        n_samples = min(3, int(audio_duration / duration))
                        for _ in range(n_samples):
                            self.samples.append((idx, -1.0, audio_duration))
                    else:
                        n_windows = int(audio_duration / duration)
                        for i in range(n_windows):
                            self.samples.append((idx, i * duration, audio_duration))
            except Exception:
                self.samples.append((idx, 0.0, 0.0))

    def __len__(self):
        return len(self.samples)

    def _apply_specaugment(self, mel_db):
        # mel_db: (n_mels, n_frames)
        mel_aug = mel_db.copy()

        for _ in range(np.random.randint(1, 3)):
            f_size = np.random.randint(5, 12)
            if f_size < mel_aug.shape[0]:
                f_start = np.random.randint(0, mel_aug.shape[0] - f_size)
                mel_aug[f_start:f_start + f_size, :] = 0.0

        for _ in range(np.random.randint(1, 3)):
            t_size = np.random.randint(10, 100)
            if t_size < mel_aug.shape[1]:
                t_start = np.random.randint(0, mel_aug.shape[1] - t_size)
                mel_aug[:, t_start:t_start + t_size] = 0.0

        return mel_aug

    def __getitem__(self, idx):
        audio_idx, start_secs, audio_duration = self.samples[idx]
        row = self.data.iloc[audio_idx]

        if start_secs < 0:
            max_start = max(0.0, audio_duration - self.duration)
            start_secs = np.random.uniform(0.0, max_start)

        try:
            y, _ = librosa.load(
                row["file_path"],
                sr=self.sr,
                mono=True,
                offset=start_secs,
                duration=self.duration,
            )
        except Exception:
            y = np.zeros(self.win_len, dtype=np.float32)

        labels = row[self.label_cols].astype(float).values

        if len(y) < self.win_len:
            y = np.pad(y, (0, self.win_len - len(y)))

        if self.augment:
            if np.random.rand() < 0.5:
                y = y * np.random.uniform(0.7, 1.3)

            if np.random.rand() < 0.4:
                noise_factor = np.random.uniform(0.001, 0.005)
                y = y + noise_factor * np.random.randn(len(y))

            if np.random.rand() < 0.15:
                rate = np.random.uniform(0.95, 1.05)
                y_stretched = librosa.effects.time_stretch(y, rate=rate)
                if len(y_stretched) >= self.win_len:
                    y = y_stretched[:self.win_len]
                else:
                    y = np.pad(y_stretched, (0, self.win_len - len(y_stretched)))

            if np.random.rand() < 0.15:
                n_steps = np.random.uniform(-2, 2)
                y = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)

            if np.random.rand() < 0.3:
                shift = np.random.randint(-self.sr // 2, self.sr // 2)
                y = np.roll(y, shift)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=1024,
            hop_length=320,
            fmin=50,
            fmax=14000,
        )
        mel_db = librosa.power_to_db(mel, ref=max(mel.max(), 1e-10))

        if self.augment and np.random.rand() < 0.5:
            mel_db = self._apply_specaugment(mel_db)

        # Shape: (1, time_frames, mel_bins) — PANNs format; BN0 in the model normalises
        x = mel_db.T[np.newaxis, :, :]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
