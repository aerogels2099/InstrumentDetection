# Instrument Detection

A web application for detecting musical instruments in audio files using a pre-trained deep learning model.

## Overview

Uploads an audio file and returns a list of detected instruments with confidence scores. The threshold for detection can be adjusted live in the UI without re-uploading the file.

The model is a **CNN14** backbone from [PANNs (Pre-trained Audio Neural Networks)](https://github.com/qiuqiangkong/audioset_tagging_cnn), pre-trained on AudioSet and fine-tuned for multi-label instrument classification. Audio is processed as a log-mel spectrogram (32 kHz, 64 mel bins) and classified across 13 instrument classes.

## Structure

```
instrument-detection/
├── backend/
│   ├── app.py                  # FastAPI server
│   ├── model.py                # CNN14 architecture
│   ├── instrument_inference.py # Inference pipeline
│   ├── instrument_labels.py    # Instrument class names
│   ├── instrument_cnn.py       # Training entry point
│   ├── dataset.py              # Dataset + augmentation
│   ├── training.py             # Training loop + evaluation
│   ├── loss.py                 # Asymmetric loss
│   ├── download_panns.py       # Download PANNs checkpoint
│   ├── generate_csv.py         # Build train/val CSV from audio folder
│   └── requirements.txt
└── frontend/
    └── src/App.jsx             # React/Vite UI
```

## Setup

### Backend

```bash
cd instrument-detection/backend
pip install -r requirements.txt
```

### Frontend

```bash
cd instrument-detection/frontend
npm install
```

## Training

1. Download the PANNs CNN14 pre-trained checkpoint (~120 MB):
   ```bash
   python download_panns.py
   ```

2. Prepare your dataset. Audio files should be organized by instrument label, then run:
   ```bash
   python generate_csv.py
   ```
   This produces `data/train.csv` and `data/val.csv`.

3. Train the model:
   ```bash
   python instrument_cnn.py
   ```
   The best checkpoint is saved as `instrument_cnn14_final.pth` (excluded from the repo — generate it by running training).

## Running

Start the backend:
```bash
cd instrument-detection/backend
uvicorn app:app --reload
```

Start the frontend (development):
```bash
cd instrument-detection/frontend
npm run dev
```

The API is also available standalone at `http://localhost:8000/api/predict`.

## Supported Instruments

Guitar, Piano, Violin, Drums, Bass, Flute, Cello, Trumpet, Saxophone, Clarinet, Voice, Organ, Synthesizer *(varies by training data)*

## Acknowledgments

Pre-trained weights from [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) by Qiuqiang Kong et al., trained on [AudioSet](https://research.google.com/audioset/).

Dynamic background inspired by [AmbientCanvasBackgrounds](https://github.com/crnacura/AmbientCanvasBackgrounds) by *crnacura*.
