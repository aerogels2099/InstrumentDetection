from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
import torch
from instrument_inference import load_model, predict_instruments, INSTRUMENTS

app = FastAPI(title="Instrument Detection API")
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "build")
app.mount("/frontend", StaticFiles(directory=frontend_path, html=True), name="frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("instrument_cnn_new.pth", device=device)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3")):
        return JSONResponse(status_code=400, content={"error": "Only WAV or MP3 files are supported."})

    temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, temp_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        detected, scores = predict_instruments(file_path, model)

        response = {
            "detected_instruments": detected,
            "scores": {inst: round(float(scores[i]), 3) for i, inst in enumerate(INSTRUMENTS)}
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return JSONResponse(content=response)

@app.get("/health")
async def health():
    return {"status": "Instrument Detection API is running"}