import os
import random
import pandas as pd
import librosa

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
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
TRAIN_SPLIT = 0.8
SEED = 42

# Cap any single label at this many files to prevent class imbalance.
# Set to None to disable capping.
MAX_PER_CLASS = 1500

random.seed(SEED)

rows_by_label = {}

for label in LABELS:
    label_dir = os.path.join(AUDIO_ROOT, label)

    if not os.path.isdir(label_dir):
        print(f"Missing folder: {label_dir}")
        continue

    label_rows = []
    for fname in os.listdir(label_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in AUDIO_EXTS:
            continue

        path = os.path.normpath(os.path.join(label_dir, fname))
        try:
            duration = librosa.get_duration(path=path)
        except Exception:
            duration = 0.0

        row = {"file_path": path, "duration": duration}
        for l in LABELS:
            row[l] = 1 if l == label else 0

        label_rows.append(row)

    random.shuffle(label_rows)
    if MAX_PER_CLASS is not None and len(label_rows) > MAX_PER_CLASS:
        print(f"  {label}: {len(label_rows)} files -> capped at {MAX_PER_CLASS}")
        label_rows = label_rows[:MAX_PER_CLASS]
    else:
        print(f"  {label}: {len(label_rows)} files")

    rows_by_label[label] = label_rows

all_rows = [row for rows in rows_by_label.values() for row in rows]
random.shuffle(all_rows)
split_idx = int(len(all_rows) * TRAIN_SPLIT)

train_df = pd.DataFrame(all_rows[:split_idx])
val_df = pd.DataFrame(all_rows[split_idx:])

train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)

print(f"\ntrain.csv: {len(train_df)} samples")
print(f"val.csv:   {len(val_df)} samples")
print(f"total:     {len(all_rows)} samples")
