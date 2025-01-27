# app.py
import os
import re
import librosa
import torch
from fastapi import (
    FastAPI, File, UploadFile, HTTPException, Depends, Form, status
)
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf
import numpy as np
import math
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.orm import Session
from database import SessionLocal, init_db
from models import Share, User, Upload
from schemas import (
    ShareInfo, UploadHistory, UserCreate, UserLogin, Token,
    UploadHistoryResponse, ShareCreate, ShareResponse
)
from auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES, verify_password,
    get_password_hash, create_access_token,
    decode_access_token
)
from datetime import timedelta
import uuid
from werkzeug.utils import secure_filename  # For filename sanitization
import logging
import aiofiles
from collections import defaultdict
from sqlalchemy import or_
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize the database
init_db()

# Configure a ThreadPoolExecutor for handling transcription tasks
executor = ThreadPoolExecutor(max_workers=3)  # Adjust based on server capacity

# Dictionary to store locks per user
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
    "https://shaz.ai",
    "https://backend.shaz.ai",
    "http://51.15.224.218:4200",
    "http://medtranscribe.fr",
    "https://medtranscribe.fr",
    "https://www.medtranscribe.fr",
    "http://localhost:4200",
    
    # Add other origins if necessary
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Authentication dependency
async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    token_data = decode_access_token(token)
    if token_data is None or token_data.username is None:
        logger.warning("Invalid or missing token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        logger.warning(f"User not found: {token_data.username}")
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

# Directory to store audio files
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Global cache for ASR pipelines
models_cache = {}

def get_asr_pipeline(model_name: str):
    """
    Retrieves the ASR pipeline from the cache or loads it if not present.

    Args:
        model_name: The name/path of the ASR model.

    Returns:
        The ASR pipeline instance.
    """
    if model_name not in models_cache:
        try:
            logger.info(f"Loading ASR model: {model_name}")
            model_instance = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            model_instance.to(torch_device)
            processor = AutoProcessor.from_pretrained(model_name)

            asr_pipeline_instance = pipeline(
                "automatic-speech-recognition",
                model=model_instance,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=30,
                batch_size=16,
                torch_dtype=torch_dtype,
                device=pipeline_device,
                generate_kwargs={"language": "french"},
            )

            models_cache[model_name] = asr_pipeline_instance
            logger.info(f"ASR model loaded and cached: {model_name}")
        except Exception as e:
            logger.error(f"Error loading ASR model '{model_name}': {e}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {e}")
    else:
        logger.info(f"ASR model retrieved from cache: {model_name}")

    return models_cache[model_name]

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
    model: str = Form("openai/whisper-large-v3-turbo"),  # Default model
    db: Session = Depends(get_db),
):
    """
    Endpoint to upload an audio file and receive its transcription.
    Ensures that each user can only have one transcription request at a time.
    Automatically selects the model based on audio duration.
    """
    logger.info(f"User {user.username} is uploading file: {file.filename} using model: {model}")

    # Get user-specific lock
    user_lock = user_locks[user.id]

    # Acquire the lock
    async with user_lock:
        logger.info(f"Acquired lock for user {user.username}")

        # Generate unique filename using UUID
        unique_id = uuid.uuid4().hex
        file_extension = os.path.splitext(file.filename)[1].lower()
        new_filename = f"{unique_id}{file_extension}"

        # Extract main content type
        content_type_main = file.content_type.split(";")[0].strip()

        # Supported audio types
        supported_types = [
            "audio/wav", "audio/mpeg", "audio/mp3", "audio/x-wav",
            "audio/webm", "audio/ogg", "audio/flac", "audio/m4a",
            "audio/x-m4a", "audio/mp4", "audio/aac"
        ]

        if content_type_main not in supported_types:
            logger.warning(f"Unsupported file type: {content_type_main}")
            raise HTTPException(status_code=400, detail="Unsupported audio file format.")

        # Save uploaded audio file asynchronously
        file_path = os.path.join(AUDIO_DIR, new_filename)
        try:
            async with aiofiles.open(file_path, "wb") as out_file:
                content = await file.read()
                await out_file.write(content)
            logger.info(f"File saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            raise HTTPException(status_code=500, detail="Failed to save the uploaded file.")

        # Convert to WAV if necessary
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
                raise HTTPException(status_code=400, detail=f"Error converting audio: {e}")

        # Determine the duration of the audio
        try:
            audio_data, samplerate = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=audio_data, sr=samplerate)
            logger.info(f"Audio duration: {duration:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading audio for duration calculation: {e}")
            raise HTTPException(status_code=500, detail="Failed to process the audio file.")

        # Select model based on duration
        if duration > 120:
            selected_model = "openai/whisper-large-v3"
            logger.info(f"Audio longer than 2 minutes. Using model: {selected_model}")
        else:
            selected_model = model
            logger.info(f"Audio 2 minutes or shorter. Using model: {selected_model}")

        # Retrieve ASR pipeline from cache
        try:
            asr_pipeline = get_asr_pipeline(selected_model)
        except HTTPException as e:
            raise e

        # Process transcription in a separate thread
        try:
            transcription = await asyncio.get_event_loop().run_in_executor(
                executor, process_transcription, file_path, asr_pipeline
            )
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise HTTPException(status_code=500, detail=f"Error during transcription: {e}")

        # بررسی اینکه آیا کاربر 'word' هست یا خیر
        if user.username == 'word':
            # حذف فایل‌های صوتی و ترانسکریپشن
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted audio file for user 'word': {file_path}")
                # اگر فایل ترانسکریپشن نیز ایجاد شده باشد، حذف کنید
                transcription_filename = f"{unique_id}_transcription.txt"
                transcription_path = os.path.join(AUDIO_DIR, transcription_filename)
                if os.path.exists(transcription_path):
                    os.remove(transcription_path)
                    logger.info(f"Deleted transcription file for user 'word': {transcription_path}")
            except Exception as e:
                logger.error(f"Error deleting files for user 'word': {e}")
                raise HTTPException(status_code=500, detail="Failed to delete files for user 'word'.")

            # بازگشت ترانسکریپشن بدون ذخیره در پایگاه داده
            return JSONResponse(
                content={
                    "transcription": transcription.strip(),
                    "model_used": selected_model  # Inform the user about the model used
                }
            )

        # اگر کاربر 'word' نیست، ادامه عملیات به صورت معمولی
        # Generate transcription filename
        transcription_filename = f"{unique_id}_transcription.txt"
        transcription_path = os.path.join(AUDIO_DIR, transcription_filename)

        # Save transcription asynchronously
        try:
            async with aiofiles.open(transcription_path, "w", encoding="utf-8") as f:
                await f.write(transcription.strip())
            logger.info(f"Transcription saved to {transcription_path}")
        except Exception as e:
            logger.error(f"Error saving transcription: {e}")
            raise HTTPException(status_code=500, detail="Failed to save the transcription.")

        # Save upload record
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
                status_code=500, 
                detail="Failed to save the upload record in the database."
            )

        return JSONResponse(
            content={
                "transcription": transcription.strip(),
                "transcription_file": transcription_filename,
                "upload_id": upload_record.id,
                "model_used": selected_model  # Inform the user about the model used
            }
        )

@app.post("/process-chunk/")
async def process_chunk(
    user: User = Depends(get_current_user),
    file: UploadFile = File(...),
    chunk_number: int = Form(...),
    session_id: str = Form(...),
    is_final: bool = Form(False),
    model: str = Form("openai/whisper-large-v3-turbo"),
    db: Session = Depends(get_db),
):
    """Process a single chunk of audio and return its transcription."""
    logger.info(f"Processing chunk {chunk_number} for session {session_id}")
    
    # Create session directory if it doesn't exist
    session_dir = os.path.join(AUDIO_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    # Save chunk
    chunk_filename = f"chunk_{chunk_number}.wav"
    chunk_path = os.path.join(session_dir, chunk_filename)
    
    try:
        async with aiofiles.open(chunk_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)
        logger.info(f"Chunk saved to {chunk_path}")
        
        # Process transcription
        asr_pipeline = get_asr_pipeline(model)
        transcription = await asyncio.get_event_loop().run_in_executor(
            executor, process_transcription, chunk_path, asr_pipeline
        )
        
        # Save transcription
        trans_filename = f"chunk_{chunk_number}_trans.txt"
        trans_path = os.path.join(session_dir, trans_filename)
        async with aiofiles.open(trans_path, "w", encoding="utf-8") as f:
            await f.write(transcription.strip())
            
        if is_final:
            # Combine all chunks
            combined_audio = AudioSegment.empty()
            combined_trans = []
            chunk_num = 0
            while True:
                wav_path = os.path.join(session_dir, f"chunk_{chunk_num}.wav")
                trans_path = os.path.join(session_dir, f"chunk_{chunk_num}_trans.txt")
                if not os.path.exists(wav_path):
                    break
                    
                # Combine audio
                chunk_audio = AudioSegment.from_wav(wav_path)
                combined_audio += chunk_audio
                
                # Combine transcription
                async with aiofiles.open(trans_path, "r", encoding="utf-8") as f:
                    chunk_trans = await f.read()
                    combined_trans.append(chunk_trans.strip())
                
                chunk_num += 1
            
            # Save final files
            final_audio_path = os.path.join(AUDIO_DIR, f"{session_id}.wav")
            combined_audio.export(final_audio_path, format="wav")
            
            final_trans = " ".join(combined_trans)
            final_trans_path = os.path.join(AUDIO_DIR, f"{session_id}_transcription.txt")
            async with aiofiles.open(final_trans_path, "w", encoding="utf-8") as f:
                await f.write(final_trans)
            
            # Create upload record
            if user.username != 'word':
                upload_record = Upload(
                    filename=f"{session_id}.wav",
                    transcription_filename=f"{session_id}_transcription.txt",
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
                        status_code=500,
                        detail="Failed to save the upload record in the database."
                    )
                
                return JSONResponse(
                    content={
                        "transcription": final_trans,
                        "upload_id": upload_record.id,
                        "model_used": model
                    }
                )
            else:
                return JSONResponse(
                    content={
                        "transcription": final_trans,
                        "model_used": model
                    }
                )
        
        return JSONResponse(
            content={
                "chunk_transcription": transcription.strip(),
                "chunk_number": chunk_number
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing chunk: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chunk: {e}")

def diviser_audio(audio: np.ndarray, samplerate: int = 16000, duree_max: int = 29, overlap_duration: int = 1) -> List[np.ndarray]:
    """
    Divides audio into segments with a maximum duration and specified overlap.

    :param audio: Numpy array containing audio data
    :param samplerate: Sampling rate in Hz
    :param duree_max: Maximum duration of each segment in seconds
    :param overlap_duration: Duration of overlap between segments in seconds
    :return: List of audio segments
    """
    frames_max = int(duree_max * samplerate)
    overlap_frames = int(overlap_duration * samplerate)
    total_frames = len(audio)
    segments = []
    start = 0

    while start < total_frames:
        end = start + frames_max
        segment = audio[start:end]
        segments.append(segment)
        start += frames_max - overlap_frames  # Move start with overlap

    return segments

def diviser_audio_silence(audio: np.ndarray, samplerate: int = 16000, top_db: int = 30, min_silence_duration: float = 0.5) -> List[np.ndarray]:
    """
    Divides audio into segments based on silence.

    :param audio: Numpy array containing audio data
    :param samplerate: Sampling rate in Hz
    :param top_db: The threshold (in decibels) below reference to consider as silence
    :param min_silence_duration: Minimum duration of silence to split (in seconds)
    :return: List of audio segments
    """
    intervals = librosa.effects.split(audio, top_db=top_db, hop_length=512)

    segments = []
    for start, end in intervals:
        segment = audio[start:end]
        segments.append(segment)

    return segments

def process_transcription(file_path: str, asr_pipeline) -> str:
    """
    Process audio transcription in a separate thread.

    Args:
        file_path: Path to the audio file
        asr_pipeline: The initialized ASR pipeline to use for transcription

    Returns:
        str: The transcribed text
    """
    logger.info(f"Processing transcription for {file_path}")
    try:
        audio_data, samplerate = sf.read(file_path)
        logger.info(f"Audio data loaded: {len(audio_data)} samples at {samplerate} Hz")

        if samplerate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)
            samplerate = 16000
            logger.info(f"Audio resampled to {samplerate} Hz")

        # Choose segmentation method: overlapping or silence-based
        # For this example, we'll use overlapping segments
        overlap_duration = 1  # 1 second overlap
        segments = diviser_audio(audio_data, samplerate, duree_max=29, overlap_duration=overlap_duration)
        logger.info(f"Audio divided into {len(segments)} segment(s) with overlapping.")

        transcription_complete = ""

        for idx, segment in enumerate(segments):
            try:
                logger.info(f"Processing segment {idx + 1}/{len(segments)}")
                transcription = asr_pipeline(segment)["text"]

                # If not the first segment, remove the first 'overlap_duration' seconds worth of words
                if idx > 0:
                    # Heuristic: Remove a proportion of words based on overlap_duration
                    words = transcription.split()
                    words_to_remove = max(1, int(len(words) * (overlap_duration / 29)))
                    transcription = ' '.join(words[words_to_remove:])

                transcription_complete += transcription + " "
                logger.info(f"Segment {idx + 1} transcription: {transcription}")
            except Exception as e:
                logger.error(f"Error processing segment {idx + 1}: {e}")
                raise e

        # Optionally, you can use silence-based segmentation instead
        # Uncomment the following lines to use silence-based segmentation
        """
        segments = diviser_audio_silence(audio_data, samplerate, top_db=30, min_silence_duration=0.5)
        logger.info(f"Audio divided into {len(segments)} segment(s) based on silence.")

        for idx, segment in enumerate(segments):
            try:
                logger.info(f"Processing segment {idx + 1}/{len(segments)}")
                transcription = asr_pipeline(segment)["text"]
                transcription_complete += transcription + " "
                logger.info(f"Segment {idx + 1} transcription: {transcription}")
            except Exception as e:
                logger.error(f"Error processing segment {idx + 1}: {e}")
                raise e
        """

        # Replace punctuation words with corresponding signs
        transcription_complete = remplacer_ponctuation(transcription_complete)
        logger.info("Punctuation replaced in complete transcription.")

        return transcription_complete.strip()
    except Exception as e:
        logger.error(f"Failed to process transcription for {file_path}: {e}")
        raise e

@app.get("/download-transcription/{upload_id}")
async def download_transcription(
    upload_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Endpoint to download the transcription of a specific upload.
    Allows access for both owners and users with shared access."""
    
    # Check user access
    upload = db.query(Upload).filter(
        Upload.id == upload_id
    ).filter(
        or_(
            Upload.owner_id == current_user.id,
            Upload.shares.any(Share.user_id == current_user.id)
        )
    ).first()

    if not upload:
        logger.warning(f"Transcription not found or access denied for upload_id: {upload_id}")
        raise HTTPException(
            status_code=404, 
            detail="Transcription not found or access denied"
        )

    transcription_path = os.path.join(AUDIO_DIR, upload.transcription_filename)
    if not os.path.exists(transcription_path):
        logger.error(f"Transcription file not found: {transcription_path}")
        raise HTTPException(status_code=404, detail="Transcription file not found")

    return FileResponse(
        transcription_path, 
        media_type="text/plain", 
        filename=f"Patient_{upload_id}.txt"
    )

@app.get("/download-audio/{upload_id}")
async def download_audio(
    upload_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Endpoint to download the audio file.
    Allows access for both owners and users with shared access."""
    
    # Check user access
    upload = db.query(Upload).filter(
        Upload.id == upload_id
    ).filter(
        or_(
            Upload.owner_id == current_user.id,
            Upload.shares.any(Share.user_id == current_user.id)
        )
    ).first()

    if not upload:
        logger.warning(f"Audio file not found or access denied for upload_id: {upload_id}")
        raise HTTPException(
            status_code=404, 
            detail="Audio file not found or access denied"
        )

    audio_path = os.path.join(AUDIO_DIR, upload.filename)
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"Patient_{upload_id}.wav"
    )

@app.get("/history/", response_model=UploadHistoryResponse)
async def get_history(
    include_archived: bool = False,
    include_shared: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the upload history of the user with optional filters."""
    query = db.query(Upload)
    
    if include_shared:
        query = query.filter(
            or_(
                Upload.owner_id == current_user.id,
                Upload.shares.any(Share.user_id == current_user.id)
            )
        )
    else:
        query = query.filter(Upload.owner_id == current_user.id)

    if not include_archived:
        query = query.filter(Upload.is_archived == False)
    
    uploads = query.order_by(Upload.upload_time.desc()).all()
    
    history = []
    for upload in uploads:
        shared_with = [
            ShareInfo(user_id=share.user_id, access_type=share.access_type)
            for share in upload.shares
        ]
        history.append({
            "upload_id": upload.id,
            "filename": upload.filename,
            "transcription_filename": upload.transcription_filename,
            "upload_time": upload.upload_time.isoformat(),
            "is_archived": upload.is_archived,
            "shared_with": shared_with,
            "owner_id": upload.owner_id
        })
    
    return {
        "history": history,
        "current_user_id": current_user.id
    }

@app.get("/get-transcription/{upload_id}")
async def get_transcription(
    upload_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Endpoint to retrieve the transcription text of a specific record."""
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    
    if not upload:
        logger.warning(f"Upload not found: {upload_id}")
        raise HTTPException(status_code=404, detail="Record not found")
        
    # Check if the user has access
    has_access = False
    is_editor = False
    if upload.owner_id == current_user.id:
        has_access = True
        is_editor = True  # Owner has editor access
    else:
        share = db.query(Share).filter(
            Share.upload_id == upload_id,
            Share.user_id == current_user.id
        ).first()
        if share:
            has_access = True
            if share.access_type == 'editor':
                is_editor = True
    
    if not has_access:
        logger.warning(f"Access denied for user {current_user.username} to upload {upload_id}")
        raise HTTPException(status_code=403, detail="Access denied")
    
    transcription_path = os.path.join(AUDIO_DIR, upload.transcription_filename)
    if not os.path.exists(transcription_path):
        logger.error(f"Transcription file not found: {transcription_path}")
        raise HTTPException(status_code=404, detail="Transcription file not found")
    
    try:
        async with aiofiles.open(transcription_path, "r", encoding="utf-8") as f:
            transcription = await f.read()
        logger.info(f"Transcription retrieved for upload_id: {upload_id}")
    except Exception as e:
        logger.error(f"Error reading transcription file: {e}")
        raise HTTPException(status_code=500, detail="Failed to read the transcription file")
    
    return {"transcription": transcription, "is_editor": is_editor}

def remplacer_ponctuation(transcription: str) -> str:
   
    PUNCTUATION_MAP = {
        "point": ".",
        "virgule": ",",
        "nouvelle ligne": "\n",
        "deux points": ":",
        "2 points": ":",
        "à la ligne": "\n",
        "sur la ligne": "\n",
        "point d'exclamation": "!",
        "point d'interrogation": "?",
        "deux points": ":",
        "point-virgule": ";",
        "trait d'union": "-",
        "parenthèse ouvrante": "(",
        "parenthèse fermante": ")",
        "guillemets ouvrants": "«",
        "guillemets fermants": "»",
        "apostrophe": "'",
        "barre oblique": "/",
        "barre oblique inversée": "\\",
        "astérisque": "*",
        "tilde": "~",
        "dièse": "#",
        "dollar": "$",
        "pourcentage": "%",
        "arobase": "@",
        "égal": "=",
        "crochet ouvrant": "[",
        "crochet fermant": "]",
        "accolade ouvrante": "{",
        "accolade fermante": "}",
        "entre parenthèses": "(",
        "fermez la parenthèse": ")",
        "ouvrez la parenthèse": "(",
        "tiret du 6": "-",
        "à la ligne:": "\n ",
        "à la ligne.": "\n ",
        "à la ligne .": "\n ",
        ". à la ligne": "\n ",
        "points de suspension": "...",
        "double point": "..",
        ". .": ".",
        ".s": ".",
        "retour chariot": "\r\n",
        "tabulation": "\t",
        "espace insécable": " ",
        "puce": "•",
        "tiret cadratin": "—",
        "tiret demi-cadratin": "–",
        "plus ou moins": "±",
        "multiplié": "×",
        "divisé": "÷",
        "degré": "°",
        "micro": "µ",
        "paragraphe": "§",
        "copyright": "©️",
        "guillemets anglais ouvrants": "\"",
        "guillemets anglais fermants": "\"",
        "guillemets simples ouvrants": "'",
        "guillemets simples fermants": "'",
        "flèche droite": "→",
        "flèche gauche": "←",
        "flèche haut": "↑",
        "flèche bas": "↓",
        "inférieur ou égal": "≤",
        "supérieur ou égal": "≥",
        "différent": "≠",
        "environ": "≈",
        "pour mille": "‰",
        "exposant un": "¹",
        "exposant deux": "²",
        "exposant trois": "³",
    }
    
    
    sorted_punctuations = sorted(PUNCTUATION_MAP.keys(), key=len, reverse=True)
    
    
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted_punctuations) + r')\b', re.IGNORECASE)
    
    
    def replacer(match):
        word = match.group(0).lower()
        return PUNCTUATION_MAP.get(word, match.group(0))
    
    
    transcription = pattern.sub(replacer, transcription)
    
    
    ADDITIONAL_REPLACEMENTS = {
        ", .": ",",
        ". .": ".",
        "..": ".",
        ". ..": ".",
        ". . .": "...",
        "\n.": "\n ",
        "\n .": "\n ",
        "\n ,": "\n ",
        "\n,": "\n ",
        ", ..": ",",
        ":.": ":",
        ",.": ".",
        ", :": ":",
    }
    
   
    for pattern, replacement in ADDITIONAL_REPLACEMENTS.items():
        transcription = transcription.replace(pattern, replacement)
    
    return transcription



@app.delete("/history/{upload_id}")
async def delete_upload(
    upload_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a specific upload and its associated files."""
    upload = (
        db.query(Upload)
        .filter(Upload.id == upload_id, Upload.owner_id == current_user.id)
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

    # Delete associated shares
    try:
        shares = db.query(Share).filter(Share.upload_id == upload_id).all()
        for share in shares:
            db.delete(share)
        db.commit()
        logger.info(f"Deleted {len(shares)} share(s) for upload_id: {upload_id}")
    except Exception as e:
        logger.error(f"Error deleting shares: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete associated shares.")

    # Delete database record
    try:
        db.delete(upload)
        db.commit()
        logger.info(f"Deleted upload record: {upload_id}")
    except Exception as e:
        logger.error(f"Database deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete upload record.")

    return {"message": "Upload deleted successfully"}

@app.put("/history/{upload_id}", response_model=UploadHistory)
async def update_transcription(
    upload_id: int,
    transcription: str = Form(...),
    filename: str = Form(None),  # New optional parameter
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update transcription and optionally filename for a specific record."""
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    
    if not upload:
        logger.warning(f"Upload not found: {upload_id}")
        raise HTTPException(status_code=404, detail="Record not found")
    
    # Check permissions
    is_editor = False
    if upload.owner_id == current_user.id:
        is_editor = True
    else:
        share = db.query(Share).filter(
            Share.upload_id == upload_id,
            Share.user_id == current_user.id,
            Share.access_type == 'editor'
        ).first()
        if share:
            is_editor = True
    
    if not is_editor:
        logger.warning(f"User {current_user.username} lacks permission to edit upload {upload_id}")
        raise HTTPException(
            status_code=403, 
            detail="You do not have permission to modify this transcription"
        )
    
    # Update transcription file
    transcription_path = os.path.join(AUDIO_DIR, upload.transcription_filename)
    if not os.path.exists(transcription_path):
        logger.error(f"Transcription file not found: {transcription_path}")
        raise HTTPException(
            status_code=404, 
            detail="Transcription file not found"
        )
    
    try:
        async with aiofiles.open(transcription_path, "w", encoding="utf-8") as f:
            await f.write(transcription)
        logger.info(f"Transcription updated for upload_id: {upload_id}")
        
        # Update filename if provided
        if filename:
            # Sanitize the filename
            safe_filename = secure_filename(filename)
            upload.filename = safe_filename
            db.commit()
            logger.info(f"Filename updated to: {safe_filename}")
    except Exception as e:
        logger.error(f"Error updating transcription or filename: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to update transcription or filename"
        )
    
    # Update database if needed
    db.refresh(upload)
    
    return {
        "upload_id": upload.id,
        "filename": upload.filename,
        "transcription_filename": upload.transcription_filename,
        "upload_time": upload.upload_time.isoformat(),
        "is_archived": upload.is_archived,
        "shared_with": [
            ShareInfo(user_id=share.user_id, access_type=share.access_type)
            for share in upload.shares
        ],
        "owner_id": upload.owner_id
    }

@app.get("/stream-audio/{upload_id}")
async def stream_audio(
    upload_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Stream an audio file with proper headers for HTML5 audio player.
    Allows access for both owners and users with shared access."""
    
    # Check user access
    upload = db.query(Upload).filter(
        Upload.id == upload_id
    ).filter(
        or_(
            Upload.owner_id == current_user.id,
            Upload.shares.any(Share.user_id == current_user.id)
        )
    ).first()

    if not upload:
        logger.warning(f"Audio file not found or access denied for upload_id: {upload_id}")
        raise HTTPException(
            status_code=404, 
            detail="Audio file not found or access denied"
        )

    audio_path = os.path.join(AUDIO_DIR, upload.filename)
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        audio_path,
        media_type="audio/wav",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="{upload.filename}"',
        }
    )

@app.post("/toggle-archive/{upload_id}")
async def toggle_archive_status(
    upload_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Toggle the archive status of an upload."""
    # Check user access
    upload = db.query(Upload).filter(
        Upload.id == upload_id
    ).filter(
        or_(
            Upload.owner_id == current_user.id,
            Upload.shares.any(Share.user_id == current_user.id)
        )
    ).first()
    
    if not upload:
        logger.warning(f"Upload not found or access denied: {upload_id} by user {current_user.username}")
        raise HTTPException(
            status_code=404, 
            detail="Upload not found or you don't have permission to access it"
        )
    
    try:
        upload.is_archived = not upload.is_archived
        db.commit()
        logger.info(f"Archive status toggled for upload {upload_id} by user {current_user.id}")
        return {"message": "Archive status updated", "is_archived": upload.is_archived}
    except Exception as e:
        db.rollback()
        logger.error(f"Error toggling archive status: {e}")
        raise HTTPException(status_code=500, detail="Failed to update archive status")

@app.get("/users/")
async def get_users(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get list of all users except current user and exclude user 'word'"""
    users = db.query(User).filter(
        User.id != current_user.id,
        User.username != 'word'  # اضافه کردن شرط برای حذف کاربر 'word'
    ).all()
    return {
        "users": [{"id": user.id, "username": user.username} for user in users]
    }

@app.post("/share/{upload_id}/user/", response_model=ShareResponse)
async def share_with_user(
    upload_id: int,
    share: ShareCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Share a record with a user specifying the type of access.
    """
    # چک کردن اینکه آیا رکورد متعلق به کاربر جاری است
    upload = db.query(Upload).filter(
        Upload.id == upload_id,
        Upload.owner_id == current_user.id
    ).first()
    
    if not upload:
        logger.warning(f"Upload not found or not owned by user {current_user.username}: {upload_id}")
        raise HTTPException(status_code=404, detail="Record not found")

    # چک کردن اینکه آیا کاربر هدف وجود دارد
    target_user = db.query(User).filter(User.id == share.user_id).first()
    if not target_user:
        logger.warning(f"Target user not found: {share.user_id}")
        raise HTTPException(status_code=404, detail="Target user not found")
    
    # اعتبارسنجی نوع دسترسی
    if share.access_type not in ['viewer', 'editor']:
        logger.warning(f"Invalid access type: {share.access_type}")
        raise HTTPException(status_code=400, detail="Invalid access type")
    
    # چک کردن اینکه آیا اشتراک قبلاً وجود دارد
    existing_share = db.query(Share).filter(
        Share.upload_id == upload_id,
        Share.user_id == share.user_id
    ).first()
    
    if existing_share:
        existing_share.access_type = share.access_type  # بروزرسانی نوع دسترسی
        logger.info(f"Updated existing share for user {share.user_id} on upload {upload_id}")
    else:
        new_share = Share(
            upload_id=upload_id,
            user_id=share.user_id,
            access_type=share.access_type
        )
        db.add(new_share)
        logger.info(f"Created new share for user {share.user_id} on upload {upload_id}")
    
    try:
        db.commit()
        db.refresh(existing_share or new_share)
    except Exception as e:
        db.rollback()
        logger.error(f"Error sharing upload: {e}")
        raise HTTPException(status_code=500, detail="Failed to share upload")
    
    return existing_share or new_share

@app.delete("/share/{upload_id}/user/{user_id}")
async def remove_share(
    upload_id: int,
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Remove the sharing of a record with a user.
    """
    # چک کردن اینکه آیا رکورد متعلق به کاربر جاری است
    upload = db.query(Upload).filter(
        Upload.id == upload_id,
        Upload.owner_id == current_user.id
    ).first()
    
    if not upload:
        logger.warning(f"Upload not found or not owned by user {current_user.username}: {upload_id}")
        raise HTTPException(status_code=404, detail="Record not found")
    
    # یافتن اشتراک
    share = db.query(Share).filter(
        Share.upload_id == upload_id,
        Share.user_id == user_id
    ).first()
    
    if not share:
        logger.warning(f"Share not found for user {user_id} on upload {upload_id}")
        raise HTTPException(status_code=404, detail="Share not found")
    
    db.delete(share)
    try:
        db.commit()
        logger.info(f"Removed share for user {user_id} on upload {upload_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Error removing share: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove share")
    
    return {"message": "Share access removed successfully"}
