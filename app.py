# app.py
import os
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
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
from schemas import UserCreate, UserLogin, Token, UploadHistoryResponse
from auth import ACCESS_TOKEN_EXPIRE_MINUTES, verify_password, get_password_hash, create_access_token, decode_access_token
from datetime import timedelta
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
    "http://51.15.224.218:4200", 
     "http://medtranscribe.fr",
    "https://medtranscribe.fr",
    "https://www.medtranscribe.fr",
    "http://localhost:4200",
     # Your Angular app's address
    # Add other origins if necessary
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Authentication dependency
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    token_data = decode_access_token(token)
    if token_data is None or token_data.username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

# Device configuration for PyTorch and Transformers pipeline
if torch.cuda.is_available():
    torch_device = "cuda:0"
    pipeline_device = 0  # GPU index for Transformers pipeline
    torch_dtype = torch.float16
    logger.info("Using CUDA for both PyTorch and Transformers pipeline.")
elif torch.backends.mps.is_available():
    torch_device = "mps"
    pipeline_device = -1  # Transformers pipeline does not support MPS
    torch_dtype = torch.float32
    logger.info("Using MPS for PyTorch and CPU for Transformers pipeline.")
else:
    torch_device = "cpu"
    pipeline_device = -1  # CPU for Transformers pipeline
    torch_dtype = torch.float32
    logger.info("Using CPU for both PyTorch and Transformers pipeline.")

# Model identifier for Whisper large-v3-turbo
model_id = "openai/whisper-large-v3-turbo"

# Load model and processor
logger.info("Loading model and processor...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(torch_device)
processor = AutoProcessor.from_pretrained(model_id)
logger.info("Model and processor loaded.")

# Create ASR pipeline with chunked processing
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,  # Optimal chunk length for large-v3-turbo
    batch_size=16,      # Adjust based on your device's capabilities
    torch_dtype=torch_dtype,
    device=pipeline_device,
    generate_kwargs={"language": "french"}  # تنظیمات پیش‌فرض زبان
)

# Directory to store audio files
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=4)  # Adjust as needed

@app.post("/register/", response_model=Token)
def register(user: UserCreate, db: Session = Depends(get_db)):
    """
    Endpoint to register a new user.
    """
    logger.info(f"Attempting to register user: {user.username}")
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        logger.warning(f"User registration failed: {user.username} already exists.")
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    logger.info(f"User registered successfully: {user.username}")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": new_user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/token/", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Endpoint to authenticate a user and return a JWT token.
    """
    logger.info(f"User attempting to login: {form_data.username}")
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user:
        logger.warning(f"Login failed: {form_data.username} does not exist.")
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    if not verify_password(form_data.password, user.hashed_password):
        logger.warning(f"Login failed: Incorrect password for user {form_data.username}.")
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    logger.info(f"User logged in successfully: {form_data.username}")
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload-audio/")
async def upload_audio(
    user: User = Depends(get_current_user),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Endpoint to upload an audio file and receive its transcription.
    """
    logger.info(f"User {user.username} is uploading file: {file.filename}")
    
    # Get the next patient number
    last_upload = db.query(Upload).filter(
        Upload.filename.like("patient_%")
    ).order_by(Upload.id.desc()).first()
    
    if last_upload:
        try:
            last_number = int(last_upload.filename.split('_')[1].split('.')[0])
            new_number = last_number + 1
        except (IndexError, ValueError):
            new_number = 1
    else:
        new_number = 1
    
    # Create new patient filename
    file_extension = os.path.splitext(file.filename)[1].lower()
    new_filename = f"patient_{new_number}{file_extension}"
    
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
        "audio/m4a",
        "audio/x-m4a",
        "audio/mp4",
        "audio/aac",
    ]
    
    # Check if the type is supported
    if content_type_main not in supported_types:
        logger.warning(f"Unsupported file type: {content_type_main}")
        raise HTTPException(status_code=400, detail="Unsupported audio file format.")
    
    # Save the uploaded audio file with the new patient filename
    file_path = os.path.join(AUDIO_DIR, new_filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    logger.info(f"File saved to {file_path}")
    
    # Convert to WAV if necessary
    if content_type_main != "audio/wav":
        try:
            audio = AudioSegment.from_file(file_path)
            # Create WAV filename maintaining the patient number
            wav_filename = f"patient_{new_number}.wav"
            wav_path = os.path.join(AUDIO_DIR, wav_filename)
            audio = audio.set_channels(1)  # Convert to mono
            audio.export(wav_path, format="wav")
            os.remove(file_path)  # Remove original file
            file_path = wav_path
            new_filename = wav_filename  # Update filename
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
    
    # Generate transcription filename maintaining the patient number format
    transcription_filename = f"patient_{new_number}_transcription.txt"
    transcription_path = os.path.join(AUDIO_DIR, transcription_filename)
    
    # Save the transcription to a text file
    with open(transcription_path, "w", encoding="utf-8") as f:
        f.write(transcription.strip())
    logger.info(f"Transcription saved to {transcription_path}")
    
    # Save upload record to the database
    upload_record = Upload(
        filename=new_filename,  # Saved audio filename with patient number
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
    segments = diviser_audio(audio_data, samplerate, duree_max=29)
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
async def download_transcription(upload_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Endpoint to download the transcription of a specific upload.
    """
    upload_record = db.query(Upload).filter(Upload.id == upload_id, Upload.user_id == current_user.id).first()
    if not upload_record:
        logger.warning(f"Transcription record not found for upload_id: {upload_id} and user: {current_user.username}")
        raise HTTPException(status_code=404, detail="Transcription file not found.")
    
    transcription_path = os.path.join(AUDIO_DIR, upload_record.transcription_filename)
    if not os.path.exists(transcription_path):
        logger.warning(f"Transcription file not found: {upload_record.transcription_filename}")
        raise HTTPException(status_code=404, detail="Transcription file not found.")
    
    logger.info(f"Transcription file found: {upload_record.transcription_filename}")
    return FileResponse(transcription_path, media_type='text/plain', filename=upload_record.transcription_filename)

@app.get("/download-audio/{upload_id}")
async def download_audio(upload_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Endpoint to download the original audio file of a specific upload.
    """
    upload_record = db.query(Upload).filter(Upload.id == upload_id, Upload.user_id == current_user.id).first()
    if not upload_record:
        logger.warning(f"Audio record not found for upload_id: {upload_id} and user: {current_user.username}")
        raise HTTPException(status_code=404, detail="Audio file not found.")
    
    audio_path = os.path.join(AUDIO_DIR, upload_record.filename)
    if not os.path.exists(audio_path):
        logger.warning(f"Audio file not found: {upload_record.filename}")
        raise HTTPException(status_code=404, detail="Audio file not found.")
    
    logger.info(f"Audio file found: {upload_record.filename}")
    return FileResponse(audio_path, media_type='audio/wav', filename=upload_record.filename)

@app.get("/history/", response_model=UploadHistoryResponse)
async def get_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Endpoint to retrieve the upload history of the authenticated user.
    """
    uploads = db.query(Upload).filter(Upload.user_id == current_user.id).all()
    history = []
    for upload in uploads:
        history.append({
            "upload_id": upload.id,
            "filename": upload.filename,
            "transcription_filename": upload.transcription_filename,
            "upload_time": upload.upload_time.isoformat()
        })
    
    logger.info(f"Retrieved history for user: {current_user.username}, total uploads: {len(history)}")
    return {"history": history}

@app.get("/get-transcription/{upload_id}")
async def get_transcription(upload_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Endpoint to retrieve the transcription text of a specific upload.
    """
    upload_record = db.query(Upload).filter(Upload.id == upload_id, Upload.user_id == current_user.id).first()
    if not upload_record:
        logger.warning(f"Transcription record not found for upload_id: {upload_id} and user: {current_user.username}")
        raise HTTPException(status_code=404, detail="Transcription not found.")
    
    transcription_path = os.path.join(AUDIO_DIR, upload_record.transcription_filename)
    if not os.path.exists(transcription_path):
        logger.warning(f"Transcription file not found: {upload_record.transcription_filename}")
        raise HTTPException(status_code=404, detail="Transcription file not found.")
    
    with open(transcription_path, "r", encoding="utf-8") as f:
        transcription = f.read()
    
    return {"transcription": transcription}

def diviser_audio(audio, samplerate=16000, duree_max=29):
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

@app.delete("/history/{upload_id}")
async def delete_upload(
    upload_id: int, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """Delete a specific upload and its associated files."""
    upload = db.query(Upload).filter(Upload.id == upload_id, Upload.user_id == current_user.id).first()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    # Delete associated files
    audio_path = os.path.join(AUDIO_DIR, upload.filename)
    transcription_path = os.path.join(AUDIO_DIR, upload.transcription_filename)
    
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(transcription_path):
            os.remove(transcription_path)
    except Exception as e:
        logger.error(f"Error deleting files: {e}")
    
    # Delete database record
    db.delete(upload)
    db.commit()
    
    return {"message": "Upload deleted successfully"}

@app.put("/history/{upload_id}")
async def update_transcription(
    upload_id: int,
    transcription: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update the transcription text for a specific upload."""
    upload = db.query(Upload).filter(Upload.id == upload_id, Upload.user_id == current_user.id).first()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    # Update transcription file
    transcription_path = os.path.join(AUDIO_DIR, upload.transcription_filename)
    with open(transcription_path, "w", encoding="utf-8") as f:
        f.write(transcription)
    
    return {"message": "Transcription updated successfully"}