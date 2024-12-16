# app.py
import os
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import JSONResponse, FileResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf
import numpy as np
import math
from fastapi.middleware.cors import CORSMiddleware
import librosa
from pydub import AudioSegment
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.orm import Session
from database import SessionLocal, init_db
from models import User, Upload
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename  # For filename sanitization
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize the database
init_db()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# CORS configuration
origins = [
    "http://localhost:4200",  # Your Angular app's address
    # Add other origins if necessary
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device configuration for PyTorch and Transformers pipeline
if torch.cuda.is_available():
    torch_device = "cuda:0"
    pipeline_device = 0  # GPU index for Transformers pipeline
    logger.info("Using CUDA for both PyTorch and Transformers pipeline.")
elif torch.backends.mps.is_available():
    torch_device = "mps"
    pipeline_device = -1  # Transformers pipeline does not support MPS
    logger.info("Using MPS for PyTorch and CPU for Transformers pipeline.")
else:
    torch_device = "cpu"
    pipeline_device = -1  # CPU for Transformers pipeline
    logger.info("Using CPU for both PyTorch and Transformers pipeline.")

# Model identifier
model_id = "openai/whisper-large-v3"

# Load model and processor
logger.info("Loading model and processor...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch.float16 if torch_device.startswith("cuda") else torch.float32, 
    low_cpu_mem_usage=True, use_safetensors=True
)
model.to(torch_device)
processor = AutoProcessor.from_pretrained(model_id)
logger.info("Model and processor loaded.")

# Create ASR pipeline
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=pipeline_device,
    generate_kwargs={"language": "french"}
)

# Directory to store audio files
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=4)  # Adjust as needed

@app.post("/upload-audio/")
async def upload_audio(
    user_id: int = Form(...), 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    """
    Endpoint to upload an audio file and receive its transcription.
    """
    logger.info(f"Received file: {file.filename}, content_type: {file.content_type}, from user_id: {user_id}")
    
    # Sanitize the original filename
    original_filename = secure_filename(file.filename)
    
    # Check if user exists, else create
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        user = User(id=user_id, username=f"user_{user_id}")
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"Created new user with id: {user_id}")
    
    # Extract the main content type
    content_type_main = file.content_type.split(';')[0].strip()
    
    # Supported audio types
    supported_types = [
        "audio/wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/x-wav",
        "audio/webm",
        "audio/ogg",
        "audio/flac",
    ]
    
    # Check if the type is supported
    if content_type_main not in supported_types:
        logger.warning("Unsupported file type.")
        raise HTTPException(status_code=400, detail="Unsupported audio file format.")
    
    # Save the uploaded audio file
    file_path = os.path.join(AUDIO_DIR, original_filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    logger.info(f"File saved to {file_path}")
    
    # Convert to WAV if necessary
    if content_type_main != "audio/wav":
        try:
            audio = AudioSegment.from_file(file_path)
            # Append a UUID to ensure unique transcription filename
            unique_id = uuid.uuid4().hex
            wav_filename = f"{os.path.splitext(original_filename)[0]}_{unique_id}.wav"
            wav_path = os.path.join(AUDIO_DIR, wav_filename)
            audio = audio.set_channels(1)  # Convert to mono
            audio.export(wav_path, format="wav")
            os.remove(file_path)  # Remove original file
            file_path = wav_path
            original_filename = wav_filename  # Update filename
            logger.info(f"Converted file to {wav_path}")
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            raise HTTPException(status_code=400, detail=f"Error converting audio: {e}")
    
    # Process transcription asynchronously
    try:
        transcription = await asyncio.get_event_loop().run_in_executor(executor, process_transcription, file_path)
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Error during transcription: {e}")
    
    # Generate a unique transcription filename using UUID
    transcription_unique_id = uuid.uuid4().hex
    transcription_filename = f"{os.path.splitext(os.path.basename(original_filename))[0]}_transcription_{transcription_unique_id}.txt"
    transcription_path = os.path.join(AUDIO_DIR, transcription_filename)
    
    # Save the transcription to a text file
    with open(transcription_path, "w", encoding="utf-8") as f:
        f.write(transcription.strip())
    logger.info(f"Transcription saved to {transcription_path}")
    
    # Save upload record to the database
    upload_record = Upload(
        filename=original_filename,  # Saved audio filename
        transcription_filename=transcription_filename,
        owner=user
    )
    db.add(upload_record)
    try:
        db.commit()
        db.refresh(upload_record)
        logger.info(f"Upload record created with id: {upload_record.id}")
    except Exception as e:
        logger.error(f"Database commit failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to save upload record to the database.")
    
    return JSONResponse(content={
        "transcription": transcription.strip(),
        "transcription_file": transcription_filename,
        "upload_id": upload_record.id
    })

def process_transcription(file_path):
    """
    Blocking function to process audio transcription.
    """
    logger.info(f"Processing transcription for {file_path}")
    # Load audio with SoundFile
    audio_data, samplerate = sf.read(file_path)
    logger.info(f"Audio data loaded: {len(audio_data)} samples at {samplerate} Hz")
    
    # Resample if necessary
    if samplerate != 16000:
        try:
            audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)
            samplerate = 16000
            logger.info(f"Audio resampled to {samplerate} Hz")
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            raise e
    
    # Divide audio into 30-second segments
    segments = diviser_audio(audio_data, samplerate, duree_max=30)
    logger.info(f"Audio divided into {len(segments)} segment(s).")
    
    transcription_complete = ""
    
    for idx, segment in enumerate(segments):
        try:
            logger.info(f"Processing segment {idx + 1}/{len(segments)}")
            # Use ASR pipeline for transcription
            transcription = asr_pipeline(segment)["text"]
            transcription_complete += transcription + " "
            logger.info(f"Segment {idx + 1} transcription: {transcription}")
        except Exception as e:
            logger.error(f"Error processing segment {idx + 1}: {e}")
            raise e
    
    return transcription_complete

@app.get("/download-transcription/{upload_id}")
async def download_transcription(upload_id: int, db: Session = Depends(get_db)):
    """
    Endpoint to download the transcription of a specific upload.
    """
    upload_record = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload_record:
        logger.warning(f"Transcription record not found for upload_id: {upload_id}")
        raise HTTPException(status_code=404, detail="Transcription file not found.")
    
    transcription_path = os.path.join(AUDIO_DIR, upload_record.transcription_filename)
    if not os.path.exists(transcription_path):
        logger.warning(f"Transcription file not found: {upload_record.transcription_filename}")
        raise HTTPException(status_code=404, detail="Transcription file not found.")
    
    logger.info(f"Transcription file found: {upload_record.transcription_filename}")
    return FileResponse(transcription_path, media_type='text/plain', filename=upload_record.transcription_filename)

@app.get("/download-audio/{upload_id}")
async def download_audio(upload_id: int, db: Session = Depends(get_db)):
    """
    Endpoint to download the original audio file of a specific upload.
    """
    upload_record = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload_record:
        logger.warning(f"Audio record not found for upload_id: {upload_id}")
        raise HTTPException(status_code=404, detail="Audio file not found.")
    
    audio_path = os.path.join(AUDIO_DIR, upload_record.filename)
    if not os.path.exists(audio_path):
        logger.warning(f"Audio file not found: {upload_record.filename}")
        raise HTTPException(status_code=404, detail="Audio file not found.")
    
    logger.info(f"Audio file found: {upload_record.filename}")
    return FileResponse(audio_path, media_type='audio/wav', filename=upload_record.filename)

@app.get("/history/{user_id}")
async def get_history(user_id: int, db: Session = Depends(get_db)):
    """
    Endpoint to retrieve the upload history of a user.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        logger.warning(f"User not found with id: {user_id}")
        raise HTTPException(status_code=404, detail="User not found.")
    
    uploads = db.query(Upload).filter(Upload.user_id == user_id).all()
    history = []
    for upload in uploads:
        history.append({
            "upload_id": upload.id,
            "filename": upload.filename,
            "transcription_filename": upload.transcription_filename,
            "upload_time": upload.upload_time.isoformat()
        })
    
    logger.info(f"Retrieved history for user_id: {user_id}, total uploads: {len(history)}")
    return JSONResponse(content={"history": history})

@app.get("/get-transcription/{upload_id}")
async def get_transcription(upload_id: int, db: Session = Depends(get_db)):
    """
    Endpoint to retrieve the transcription text of a specific upload.
    """
    upload_record = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload_record:
        logger.warning(f"Transcription record not found for upload_id: {upload_id}")
        raise HTTPException(status_code=404, detail="Transcription not found.")
    
    transcription_path = os.path.join(AUDIO_DIR, upload_record.transcription_filename)
    if not os.path.exists(transcription_path):
        logger.warning(f"Transcription file not found: {upload_record.transcription_filename}")
        raise HTTPException(status_code=404, detail="Transcription file not found.")
    
    with open(transcription_path, "r", encoding="utf-8") as f:
        transcription = f.read()
    
    return {"transcription": transcription}

def diviser_audio(audio, samplerate=16000, duree_max=30):
    """
    Divides audio into segments with a maximum duration.
    
    :param audio: Numpy array containing audio data
    :param samplerate: Sampling rate in Hz
    :param duree_max: Maximum duration of each segment in seconds
    :return: List of audio segments
    """
    frames_max = int(duree_max * samplerate)
    total_frames = len(audio)
    segments = []
    nombre_segments = math.ceil(total_frames / frames_max)

    for i in range(nombre_segments):
        start = i * frames_max
        end = start + frames_max
        segment = audio[start:end]
        segments.append(segment)

    return segments
