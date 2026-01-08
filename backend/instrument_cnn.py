import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

class InstrumentDataset(Dataset):
    def __init__(
        self,
        csv_file,
        sr=22050,
        duration=3.0,
        hop=1.5,
        n_mels=128,
        augment=False,
        fixed_time_len=None
    ):
        self.data = pd.read_csv(csv_file)
        self.sr = sr
        self.win_len = int(sr * duration)
        self.hop_len = int(sr * hop)
        self.n_mels = n_mels
        self.augment = augment
        self.fixed_time_len = fixed_time_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        y1, _ = librosa.load(row["file_path"], sr=self.sr, mono=True)
        labels1 = row.iloc[1:].astype(float).values

        if self.augment and np.random.rand() < 0.5:
            j = np.random.randint(0, len(self.data))
            row2 = self.data.iloc[j]
            y2, _ = librosa.load(row2["file_path"], sr=self.sr, mono=True)
            labels2 = row2.iloc[1:].astype(float).values

            min_len = min(len(y1), len(y2))
            a, b = np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7)
            y = (a*y1[:min_len] + b*y2[:min_len]) / (a + b + 1e-6)
            label = np.clip(labels1 + labels2, 0, 1)
        else:
            y = y1
            label = labels1

        if self.augment:
            if np.random.rand() < 0.3:
                y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.9, 1.1))
            if np.random.rand() < 0.3:
                y = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=np.random.uniform(-1, 1))
            if np.random.rand() < 0.3:
                y += 0.005 * np.random.randn(len(y))

        if len(y) < self.win_len:
            y = np.pad(y, (0, self.win_len - len(y)))

        chunks = []
        start = 0
        while start + self.win_len <= len(y):
            chunk = y[start:start+self.win_len]
            chunks.append(chunk)
            start += self.hop_len
        if len(chunks) == 0:
            chunks.append(y[:self.win_len])
        y = np.concatenate(chunks)

        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel)
        delta = librosa.feature.delta(mel_db)
        delta2 = librosa.feature.delta(mel_db, order=2)
        x = np.stack([mel_db, delta, delta2])

        max_len = self.fixed_time_len if self.fixed_time_len else x.shape[2]
        if x.shape[2] < max_len:
            x = np.pad(x, ((0,0),(0,0),(0,max_len - x.shape[2])), mode="constant")
        else:
            x = x[:, :, :max_len]

        x = (x - x.mean()) / (x.std() + 1e-6)
        x = torch.tensor(x, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return x, label

class CNN(nn.Module):
    def __init__(self, num_classes=13, n_mels=128):
        super().__init__()
        self.n_mels = n_mels
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )
        self.gru = nn.GRU(
            input_size=64 * (self.n_mels // 4),
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x)                 # (B,C,F,T)
        x = x.permute(0,3,1,2)          # (B,T,C,F)
        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        return self.fc(x)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    mask = (all_labels.sum(axis=0) + all_preds.sum(axis=0)) > 0
    if mask.sum() == 0:
        return 0.0
    f1 = f1_score(all_labels[:, mask], all_preds[:, mask], average='macro', zero_division=0)
    return f1

if __name__ == "__main__":
    train_csv = "data/train.csv"
    val_csv = "data/val.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = InstrumentDataset(train_csv, augment=True, fixed_time_len=256)
    val_dataset = InstrumentDataset(val_csv, augment=False, fixed_time_len=256)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    df = pd.read_csv(train_csv)
    num_classes = df.shape[1] - 1
    model = CNN(num_classes=num_classes, n_mels=128).to(device)

    class_sums = df.iloc[:,1:].sum(axis=0) + 1e-6
    pos_weight = torch.tensor((class_sums.max() / class_sums).values, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )

    epochs = 50
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        f1 = evaluate(model, val_loader, device)
        scheduler.step(f1)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | F1: {f1:.4f}")

    torch.save(model.state_dict(), "instrument_crnn.pth")
    print("Training complete. Model saved as instrument_crnn.pth")