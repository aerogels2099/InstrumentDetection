import os
import random
import pandas as pd

LABELS = [
    "bass",
    "cello",
    "clarinet",
    "drums",
    "flute",
    "gac",
    "gel",
    "organ",
    "piano",
    "saxophone",
    "trumpet",
    "violin",
    "voice"
]

AUDIO_ROOT = "data/audio"
OUTPUT_DIR = "data"
TRAIN_SPLIT = 0.8
SEED = 42

random.seed(SEED)

rows = []

for label in LABELS:
    label_dir = os.path.join(AUDIO_ROOT, label)

    if not os.path.isdir(label_dir):
        print(f"⚠️ Missing folder: {label_dir}")
        continue

    for fname in os.listdir(label_dir):
        if not fname.lower().endswith(".wav"):
            continue

        path = os.path.join(label_dir, fname)

        row = {"file_path": path}
        for l in LABELS:
            row[l] = 1 if l == label else 0

        rows.append(row)

random.shuffle(rows)
split_idx = int(len(rows) * TRAIN_SPLIT)

train_rows = rows[:split_idx]
val_rows = rows[split_idx:]

train_df = pd.DataFrame(train_rows)
val_df = pd.DataFrame(val_rows)

train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)

print(f"📄 train.csv: {len(train_df)} samples")
print(f"📄 val.csv: {len(val_df)} samples")