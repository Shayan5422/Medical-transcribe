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
from auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    verify_password,
    get_password_hash,
    create_access_token,
    decode_access_token,
)
from datetime import timedelta
import uuid
from werkzeug.utils import secure_filename  # For filename sanitization
import logging
import aiofiles
import asyncio
from collections import defaultdict
from sqlalchemy import or_, text


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize the database
init_db()

# Configure a ThreadPoolExecutor for handling transcription tasks
executor = ThreadPoolExecutor(max_workers=1)  # Adjust the number of workers based on your server's capacity
# Dictionnaire pour stocker les verrous par utilisateur
user_locks = defaultdict(asyncio.Lock)

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
async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
):
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

# Create ASR pipeline with optimized settings
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,  # Optimal chunk length for large-v3-turbo
    batch_size=16,  # Adjust based on your device's capabilities
    torch_dtype=torch_dtype,
    device=pipeline_device,
    generate_kwargs={"language": "french"},  # تنظیمات پیش‌فرض زبان
)

# Directory to store audio files
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

@app.post("/register/", response_model=Token)
async def register(user: UserCreate, db: Session = Depends(get_db)):
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
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    """
    Endpoint to authenticate a user and return a JWT token.
    """
    logger.info(f"User attempting to login: {form_data.username}")
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user:
        logger.warning(f"Login failed: {form_data.username} does not exist.")
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    if not verify_password(form_data.password, user.hashed_password):
        logger.warning(
            f"Login failed: Incorrect password for user {form_data.username}."
        )
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
    db: Session = Depends(get_db),
):
    """
    Endpoint to upload an audio file and receive its transcription.
    Ensure that each user can only have one transcription request at a time.
    """
    logger.info(f"User {user.username} is uploading file: {file.filename}")

    # Obtenir le verrou associé à l'utilisateur
    user_lock = user_locks[user.id]

    # Tenter d'acquérir le verrou sans attendre
    if user_lock.locked():
        logger.warning(f"User {user.username} already has a transcription in progress.")
        raise HTTPException(
            status_code=429,
            detail="Vous avez déjà une transcription en cours. Veuillez réessayer plus tard.",
        )

    # Acquérir le verrou
    async with user_lock:
        logger.info(f"Acquired lock for user {user.username}")

        # Générer un nom de fichier unique en utilisant UUID pour éviter les conditions de course
        unique_id = uuid.uuid4().hex
        file_extension = os.path.splitext(file.filename)[1].lower()
        new_filename = f"{unique_id}{file_extension}"

        # Extraire le type de contenu principal
        content_type_main = file.content_type.split(";")[0].strip()

        # Types audio supportés
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

        if content_type_main not in supported_types:
            logger.warning(f"Unsupported file type: {content_type_main}")
            raise HTTPException(status_code=400, detail="Format de fichier audio non supporté.")

        # Sauvegarder le fichier audio téléchargé de manière asynchrone
        file_path = os.path.join(AUDIO_DIR, new_filename)
        try:
            async with aiofiles.open(file_path, "wb") as out_file:
                content = await file.read()
                await out_file.write(content)
            logger.info(f"File saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            raise HTTPException(status_code=500, detail="Échec de la sauvegarde du fichier téléchargé.")

        # Convertir en WAV si nécessaire
        if content_type_main != "audio/wav":
            try:
                audio = AudioSegment.from_file(file_path)
                wav_filename = f"{unique_id}.wav"
                wav_path = os.path.join(AUDIO_DIR, wav_filename)
                audio = audio.set_channels(1)
                audio.export(wav_path, format="wav")
                os.remove(file_path)
                file_path = wav_path
                new_filename = wav_filename
                logger.info(f"Converted file to {wav_path}")
            except Exception as e:
                logger.error(f"Error converting audio: {e}")
                raise HTTPException(status_code=400, detail=f"Erreur lors de la conversion de l'audio: {e}")

        # Traiter la transcription dans un thread séparé
        try:
            transcription = await asyncio.get_event_loop().run_in_executor(
                executor, process_transcription, file_path
            )
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur lors de la transcription: {e}")

        # Générer le nom de fichier de transcription
        transcription_filename = f"{unique_id}_transcription.txt"
        transcription_path = os.path.join(AUDIO_DIR, transcription_filename)

        # Sauvegarder la transcription de manière asynchrone
        try:
            async with aiofiles.open(transcription_path, "w", encoding="utf-8") as f:
                await f.write(transcription.strip())
            logger.info(f"Transcription saved to {transcription_path}")
        except Exception as e:
            logger.error(f"Error saving transcription: {e}")
            raise HTTPException(status_code=500, detail="Échec de la sauvegarde de la transcription.")

        # Sauvegarder l'enregistrement de l'upload
        upload_record = Upload(
            filename=new_filename,
            transcription_filename=transcription_filename,
            owner=user,
        )
        db.add(upload_record)
        try:
            db.commit()
            db.refresh(upload_record)
            logger.info(f"Upload record created with id: {upload_record.id}")
        except Exception as e:
            logger.error(f"Database commit failed: {e}")
            raise HTTPException(
                status_code=500, detail="Échec de la sauvegarde de l'enregistrement de l'upload dans la base de données."
            )

        return JSONResponse(
            content={
                "transcription": transcription.strip(),
                "transcription_file": transcription_filename,
                "upload_id": upload_record.id,
            }
        )


def process_transcription(file_path):
    """
    Process audio transcription in a separate thread.
    """
    logger.info(f"Processing transcription for {file_path}")
    try:
        audio_data, samplerate = sf.read(file_path)
        logger.info(f"Audio data loaded: {len(audio_data)} samples at {samplerate} Hz")

        if samplerate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)
            samplerate = 16000
            logger.info(f"Audio resampled to {samplerate} Hz")

        segments = diviser_audio(audio_data, samplerate, duree_max=29)
        logger.info(f"Audio divided into {len(segments)} segment(s).")

        transcription_complete = ""

        for idx, segment in enumerate(segments):
            try:
                logger.info(f"Processing segment {idx + 1}/{len(segments)}")
                transcription = asr_pipeline(segment)["text"]
                transcription_complete += transcription + " "
                logger.info(f"Segment {idx + 1} transcription: {transcription}")
            except Exception as e:
                logger.error(f"Error processing segment {idx + 1}: {e}")
                raise e

        return transcription_complete
    except Exception as e:
        logger.error(f"Failed to process transcription for {file_path}: {e}")
        raise e

@app.get("/download-transcription/{upload_id}")
async def download_transcription(
    upload_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Endpoint to download the transcription of a specific upload."""
    upload_record = db.query(Upload).filter(
        (Upload.id == upload_id) & 
        (
            (Upload.user_id == current_user.id) |  # User is owner
            (Upload.shared_with.contains(str([current_user.id])))  # User has shared access
        )
    ).first()

    if not upload_record:
        logger.warning(
            f"Transcription record not found or access denied for upload_id: {upload_id} and user: {current_user.username}"
        )
        raise HTTPException(status_code=404, detail="Transcription file not found or access denied.")

    transcription_path = os.path.join(AUDIO_DIR, upload_record.transcription_filename)
    if not os.path.exists(transcription_path):
        logger.warning(f"Transcription file not found: {upload_record.transcription_filename}")
        raise HTTPException(status_code=404, detail="Transcription file not found.")

    logger.info(f"Transcription file found: {upload_record.transcription_filename}")
    return FileResponse(
        transcription_path, 
        media_type="text/plain", 
        filename=upload_record.transcription_filename
    )

@app.get("/download-audio/{upload_id}")
async def download_audio(
    upload_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Endpoint to download the original audio file of a specific upload.
    """
    upload_record = (
        db.query(Upload)
        .filter(Upload.id == upload_id, Upload.user_id == current_user.id)
        .first()
    )
    if not upload_record:
        logger.warning(
            f"Audio record not found for upload_id: {upload_id} and user: {current_user.username}"
        )
        raise HTTPException(status_code=404, detail="Audio file not found.")

    audio_path = os.path.join(AUDIO_DIR, upload_record.filename)
    if not os.path.exists(audio_path):
        logger.warning(f"Audio file not found: {upload_record.filename}")
        raise HTTPException(status_code=404, detail="Audio file not found.")

    logger.info(f"Audio file found: {upload_record.filename}")
    return FileResponse(
        audio_path, media_type="audio/wav", filename=upload_record.filename
    )

@app.get("/history/", response_model=UploadHistoryResponse)
async def get_history(
    include_archived: bool = False,
    include_shared: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's upload history with optional filters"""
    # Get user's own uploads
    query = db.query(Upload).filter(Upload.user_id == current_user.id)
    
    if not include_archived:
        query = query.filter(Upload.is_archived == False)
    
    # Get uploads shared with the user
    if include_shared:
        user_id_str = str(current_user.id)
        # برای SQLite، از LIKE استفاده می‌کنیم
        shared_query = db.query(Upload).filter(
            Upload.shared_with.like(f'%{user_id_str}%')
        )
        query = query.union(shared_query)
    
    uploads = query.order_by(Upload.upload_time.desc()).all()
    
    history = []
    for upload in uploads:
        history.append({
            "upload_id": upload.id,
            "filename": upload.filename,
            "transcription_filename": upload.transcription_filename,
            "upload_time": upload.upload_time.isoformat(),
            "is_archived": upload.is_archived,
            "shared_with": upload.get_shared_users()
        })
    
    return {"history": history}

@app.get("/get-transcription/{upload_id}")
async def get_transcription(
    upload_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Endpoint to retrieve the transcription text of a specific upload."""
    # Check if user is owner or has shared access
    upload_record = db.query(Upload).filter(
        (Upload.id == upload_id) & 
        (
            (Upload.user_id == current_user.id) |  # User is owner
            (Upload.shared_with.contains(str([current_user.id])))  # User has shared access
        )
    ).first()

    if not upload_record:
        logger.warning(
            f"Transcription record not found or access denied for upload_id: {upload_id} and user: {current_user.username}"
        )
        raise HTTPException(status_code=404, detail="Transcription not found or access denied.")

    transcription_path = os.path.join(AUDIO_DIR, upload_record.transcription_filename)
    if not os.path.exists(transcription_path):
        logger.warning(f"Transcription file not found: {upload_record.transcription_filename}")
        raise HTTPException(status_code=404, detail="Transcription file not found.")

    try:
        async with aiofiles.open(transcription_path, "r", encoding="utf-8") as f:
            transcription = await f.read()
        logger.info(f"Retrieved transcription for upload_id: {upload_id}")
    except Exception as e:
        logger.error(f"Error reading transcription file: {e}")
        raise HTTPException(status_code=500, detail="Failed to read transcription file.")

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
    db: Session = Depends(get_db),
):
    """Delete a specific upload and its associated files."""
    upload = (
        db.query(Upload)
        .filter(Upload.id == upload_id, Upload.user_id == current_user.id)
        .first()
    )
    if not upload:
        logger.warning(f"Upload not found: {upload_id} for user: {current_user.username}")
        raise HTTPException(status_code=404, detail="Upload not found")

    # Delete associated files asynchronously
    audio_path = os.path.join(AUDIO_DIR, upload.filename)
    transcription_path = os.path.join(AUDIO_DIR, upload.transcription_filename)

    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.info(f"Deleted audio file: {audio_path}")
        if os.path.exists(transcription_path):
            os.remove(transcription_path)
            logger.info(f"Deleted transcription file: {transcription_path}")
    except Exception as e:
        logger.error(f"Error deleting files: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete associated files.")

    # Delete database record
    try:
        db.delete(upload)
        db.commit()
        logger.info(f"Deleted upload record: {upload_id}")
    except Exception as e:
        logger.error(f"Database deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete upload record.")

    return {"message": "Upload deleted successfully"}

@app.put("/history/{upload_id}")
async def update_transcription(
    upload_id: int,
    transcription: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update the transcription text for a specific upload."""
    upload = (
        db.query(Upload)
        .filter(Upload.id == upload_id, Upload.user_id == current_user.id)
        .first()
    )
    if not upload:
        logger.warning(f"Upload not found: {upload_id} for user: {current_user.username}")
        raise HTTPException(status_code=404, detail="Upload not found")

    transcription_path = os.path.join(AUDIO_DIR, upload.transcription_filename)
    try:
        async with aiofiles.open(transcription_path, "w", encoding="utf-8") as f:
            await f.write(transcription)
        logger.info(f"Updated transcription for upload_id: {upload_id}")
    except Exception as e:
        logger.error(f"Error updating transcription: {e}")
        raise HTTPException(status_code=500, detail="Failed to update transcription.")

    return {"message": "Transcription updated successfully"}

@app.get("/stream-audio/{upload_id}")
async def stream_audio(
    upload_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Stream an audio file with proper headers for HTML5 audio player."""
    # Check if user is owner or has shared access
    upload_record = db.query(Upload).filter(
        (Upload.id == upload_id) & 
        (
            (Upload.user_id == current_user.id) |  # User is owner
            (Upload.shared_with.contains(str([current_user.id])))  # User has shared access
        )
    ).first()

    if not upload_record:
        logger.warning(
            f"Audio record not found or access denied for upload_id: {upload_id} and user: {current_user.username}"
        )
        raise HTTPException(status_code=404, detail="Audio file not found or access denied.")

    audio_path = os.path.join(AUDIO_DIR, upload_record.filename)
    if not os.path.exists(audio_path):
        logger.warning(f"Audio file not found: {upload_record.filename}")
        raise HTTPException(status_code=404, detail="Audio file not found.")

    logger.info(f"Streaming audio file: {upload_record.filename}")
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="{upload_record.filename}"',
        },
    )
@app.post("/toggle-archive/{upload_id}")
async def toggle_archive_status(
    upload_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Toggle the archive status of an upload."""
    upload = db.query(Upload).filter(
        Upload.id == upload_id,
        Upload.user_id == current_user.id
    ).first()
    
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    upload.is_archived = not upload.is_archived
    db.commit()
    
    return {"message": "Archive status updated", "is_archived": upload.is_archived}

@app.get("/users/")
async def get_users(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get list of all users except current user"""
    users = db.query(User).filter(User.id != current_user.id).all()
    return {
        "users": [{"id": user.id, "username": user.username} for user in users]
    }

@app.post("/share/{upload_id}/user/{user_id}")
async def share_with_user(
    upload_id: int,
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Share an upload with another user"""
    # Check if upload exists and belongs to current user
    upload = db.query(Upload).filter(
        Upload.id == upload_id,
        Upload.user_id == current_user.id
    ).first()
    
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    # Check if target user exists
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Add user to shared_with list
    upload.add_shared_user(user_id)
    
    try:
        db.commit()
        logger.info(f"Upload {upload_id} shared with user {user_id}")
    except Exception as e:
        logger.error(f"Error sharing upload: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to share upload")
    
    return {"message": "Upload shared successfully"}

@app.delete("/share/{upload_id}/user/{user_id}")
async def remove_share(
    upload_id: int,
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Remove share access for a user"""
    upload = db.query(Upload).filter(
        Upload.id == upload_id,
        Upload.user_id == current_user.id
    ).first()
    
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    # Remove user from shared_with list
    upload.remove_shared_user(user_id)
    
    try:
        db.commit()
        logger.info(f"Share access removed for user {user_id} from upload {upload_id}")
    except Exception as e:
        logger.error(f"Error removing share access: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to remove share access")
    
    return {"message": "Share access removed successfully"}