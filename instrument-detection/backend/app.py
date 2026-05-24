import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.instrument_inference import load_model, predict_instruments, INSTRUMENTS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "instrument_cnn14_final.pth")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "build")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Limit simultaneous inference calls so the server stays responsive.
_predict_semaphore = asyncio.Semaphore(4)

# ---------------------------------------------------------------------------
# Model — loaded during startup lifespan
# ---------------------------------------------------------------------------

model = None
thresholds = None
device = "cpu"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model, thresholds, device

    # Remove leftover temp files from any previously crashed session.
    cutoff = time.time() - 300  # older than 5 minutes
    for fname in os.listdir(UPLOAD_DIR):
        fpath = os.path.join(UPLOAD_DIR, fname)
        if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
            try:
                os.remove(fpath)
            except OSError:
                pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    try:
        model, thresholds = load_model(MODEL_PATH, device=device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: model failed to load ({e}). /api/predict will return 503.")

    yield  # application runs here


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Instrument Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API routes  (all under /api/ so the React build can be mounted at /)
# ---------------------------------------------------------------------------

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not available. Ensure instrument_cnn14_final.pth exists."},
        )

    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
        return JSONResponse(
            status_code=400,
            content={"error": "Only WAV, MP3, FLAC, OGG, or M4A files are supported."},
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."},
        )

    temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, temp_filename)
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        async with _predict_semaphore:
            _, scores = predict_instruments(file_path, model, thresholds, device=device)

        SCORE_THRESHOLD = 0.20
        response = {
            "scores": {
                inst: round(float(scores[i]), 3)
                for i, inst in enumerate(INSTRUMENTS)
                if float(scores[i]) >= SCORE_THRESHOLD
            },
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return JSONResponse(content=response)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": device,
        "num_instruments": len(INSTRUMENTS),
    }


@app.get("/api/instruments")
async def get_instruments():
    return {"instruments": INSTRUMENTS}


# ---------------------------------------------------------------------------
# Serve the React production build at / (skipped if build dir doesn't exist)
# ---------------------------------------------------------------------------

if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
