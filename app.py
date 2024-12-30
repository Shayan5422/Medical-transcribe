# app.py
import os
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
executor = ThreadPoolExecutor(max_workers=2)  # Adjust based on server capacity

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
    "https://shaz.ai",
    "http://localhost:8000",
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
        logger.warning("Token invalide ou manquant.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        logger.warning(f"Utilisateur non trouvé: {token_data.username}")
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
    Ensure that each user can only have one transcription request at a time.
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
            raise HTTPException(status_code=400, detail="Format de fichier audio non supporté.")

        # Save uploaded audio file asynchronously
        file_path = os.path.join(AUDIO_DIR, new_filename)
        try:
            async with aiofiles.open(file_path, "wb") as out_file:
                content = await file.read()
                await out_file.write(content)
            logger.info(f"File saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            raise HTTPException(status_code=500, detail="Échec de la sauvegarde du fichier téléchargé.")

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
                raise HTTPException(status_code=400, detail=f"Erreur lors de la conversion de l'audio: {e}")

        # Load model and processor based on selected model
        try:
            model_instance = AutoModelForSpeechSeq2Seq.from_pretrained(
                model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            model_instance.to(torch_device)
            processor = AutoProcessor.from_pretrained(model)

            # Create ASR pipeline with optimized settings
            asr_pipeline = pipeline(
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
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur lors du chargement du modèle: {e}")

        # Process transcription in a separate thread
        try:
            transcription = await asyncio.get_event_loop().run_in_executor(
                executor, process_transcription, file_path, asr_pipeline
            )
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur lors de la transcription: {e}")

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
            raise HTTPException(status_code=500, detail="Échec de la sauvegarde de la transcription.")

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
                detail="Échec de la sauvegarde de l'enregistrement de l'upload dans la base de données."
            )

        return JSONResponse(
            content={
                "transcription": transcription.strip(),
                "transcription_file": transcription_filename,
                "upload_id": upload_record.id,
            }
        )

def process_transcription(file_path: str, asr_pipeline) -> str:
    """
    Process audio transcription with improved segment handling.
    
    Args:
        file_path: Path to the audio file
        asr_pipeline: The initialized ASR pipeline
        
    Returns:
        str: The transcribed text with proper word boundaries
    """
    try:
        audio_data, samplerate = sf.read(file_path)
        logger.info(f"Audio data loaded: {len(audio_data)} samples at {samplerate} Hz")

        if samplerate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)
            samplerate = 16000
            logger.info(f"Audio resampled to {samplerate} Hz")

        # Use overlapping segments
        segments = diviser_audio(audio_data, samplerate, duree_max=29, overlap_seconds=2.0)
        logger.info(f"Audio divided into {len(segments)} overlapping segments")

        transcription_complete = ""
        previous_segment_end = ""

        for idx, segment in enumerate(segments):
            try:
                logger.info(f"Processing segment {idx + 1}/{len(segments)}")
                transcription = asr_pipeline(segment)["text"].strip()
                
                # Handle overlap between segments
                if idx > 0:
                    # Find the overlapping part and merge properly
                    words = transcription.split()
                    prev_words = previous_segment_end.split()
                    
                    # Look for overlap between segments
                    overlap_found = False
                    for i in range(min(len(prev_words), len(words))):
                        if prev_words[-i:] == words[:i]:
                            transcription = " ".join(words[i:])
                            overlap_found = True
                            break
                            
                    if not overlap_found:
                        # If no clear overlap found, use a space to separate
                        transcription = " " + transcription
                
                transcription_complete += transcription
                previous_segment_end = transcription[-100:]  # Keep last portion for overlap check
                logger.info(f"Segment {idx + 1} processed successfully")
                
            except Exception as e:
                logger.error(f"Error processing segment {idx + 1}: {e}")
                raise e

        # Replace punctuation words with corresponding signs
        transcription_complete = remplacer_ponctuation(transcription_complete.strip())
        logger.info("Transcription completed with proper word boundaries")

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
    """Endpoint to download the transcription of a specific upload.
    Allows access for both owners and users with shared access."""
    
    # Vérifier l'accès de l'utilisateur
    upload = db.query(Upload).filter(
        Upload.id == upload_id
    ).filter(
        or_(
            Upload.owner_id == current_user.id,
            Upload.shares.any(Share.user_id == current_user.id)
        )
    ).first()

    if not upload:
        logger.warning(f"Transcription non trouvée ou accès refusé pour upload_id: {upload_id}")
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
    
    # Vérifier l'accès de l'utilisateur
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
    """Obtenir l'historique des uploads de l'utilisateur avec des filtres optionnels."""
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
    """Endpoint pour récupérer le texte de transcription d'un enregistrement spécifique."""
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    
    if not upload:
        logger.warning(f"Upload not found: {upload_id}")
        raise HTTPException(status_code=404, detail="Enregistrement non trouvé")
        
    # Vérifier si l'utilisateur a accès
    has_access = False
    is_editor = False
    if upload.owner_id == current_user.id:
        has_access = True
        is_editor = True  # Propriétaire a accès en tant qu'éditeur
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
        raise HTTPException(status_code=403, detail="Accès refusé")
    
    transcription_path = os.path.join(AUDIO_DIR, upload.transcription_filename)
    if not os.path.exists(transcription_path):
        logger.error(f"Transcription file not found: {transcription_path}")
        raise HTTPException(status_code=404, detail="Fichier de transcription non trouvé")
    
    try:
        async with aiofiles.open(transcription_path, "r", encoding="utf-8") as f:
            transcription = await f.read()
        logger.info(f"Transcription retrieved for upload_id: {upload_id}")
    except Exception as e:
        logger.error(f"Error reading transcription file: {e}")
        raise HTTPException(status_code=500, detail="Échec de la lecture du fichier de transcription")
    
    return {"transcription": transcription, "is_editor": is_editor}

def diviser_audio(audio: np.ndarray, samplerate: int = 16000, duree_max: int = 29, overlap_seconds: float = 2.0) -> List[np.ndarray]:
    """
    Divides audio into segments with overlap to avoid word truncation.
    
    Args:
        audio: Numpy array containing audio data
        samplerate: Sampling rate in Hz
        duree_max: Maximum duration of each segment in seconds
        overlap_seconds: Overlap duration between segments in seconds
        
    Returns:
        List of audio segments with overlap
    """
    frames_max = int(duree_max * samplerate)
    overlap_frames = int(overlap_seconds * samplerate)
    total_frames = len(audio)
    segments = []
    
    # Calculate number of segments needed with overlap
    effective_segment_length = frames_max - overlap_frames
    nombre_segments = math.ceil((total_frames - overlap_frames) / effective_segment_length)
    
    for i in range(nombre_segments):
        # Calculate start and end positions with overlap
        start = i * (frames_max - overlap_frames)
        end = min(start + frames_max, total_frames)
        
        # Create segment with overlap
        segment = audio[start:end]
        
        # Add extra silence padding if needed for last segment
        if len(segment) < frames_max:
            padding_length = frames_max - len(segment)
            segment = np.pad(segment, (0, padding_length), mode='constant')
            
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

    # Supprimer les partages associés
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
        raise HTTPException(status_code=404, detail="Enregistrement non trouvé")
    
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
            detail="Vous n'avez pas la permission de modifier cette transcription"
        )
    
    # Update transcription file
    transcription_path = os.path.join(AUDIO_DIR, upload.transcription_filename)
    if not os.path.exists(transcription_path):
        logger.error(f"Transcription file not found: {transcription_path}")
        raise HTTPException(
            status_code=404, 
            detail="Fichier de transcription non trouvé"
        )
    
    try:
        async with aiofiles.open(transcription_path, "w", encoding="utf-8") as f:
            await f.write(transcription)
        logger.info(f"Transcription updated for upload_id: {upload_id}")
        
        # Update filename if provided
        if filename:
            # Sanitize the filename
            safe_filename = secure_filename(filename)
            if safe_filename:
                # Update the custom_filename in database
                upload.custom_filename = safe_filename
                db.commit()
                logger.info(f"Filename updated for upload_id: {upload_id}")
    
    except Exception as e:
        logger.error(f"Error updating transcription or filename: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Échec de la mise à jour de la transcription ou du nom de fichier"
        )
    
    # Update database if needed
    db.refresh(upload)
    
    return {
        "upload_id": upload.id,
        "filename": upload.filename,
        "transcription_filename": upload.transcription_filename,
        "custom_filename": upload.custom_filename,  # Add this to your response model
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
    
    # Vérifier l'accès de l'utilisateur
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
    # Vérifier que l'utilisateur a accès
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
    """Get list of all users except current user"""
    users = db.query(User).filter(User.id != current_user.id).all()
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
    Partager un enregistrement avec un utilisateur spécifiant le type d'accès.
    """
    # Vérifier que l'enregistrement appartient à l'utilisateur actuel
    upload = db.query(Upload).filter(
        Upload.id == upload_id,
        Upload.owner_id == current_user.id
    ).first()
    
    if not upload:
        logger.warning(f"Upload not found or not owned by user {current_user.username}: {upload_id}")
        raise HTTPException(status_code=404, detail="Enregistrement non trouvé")

    # Vérifier que l'utilisateur cible existe
    target_user = db.query(User).filter(User.id == share.user_id).first()
    if not target_user:
        logger.warning(f"Target user not found: {share.user_id}")
        raise HTTPException(status_code=404, detail="Utilisateur cible non trouvé")
    
    # Vérifier le type d'accès
    if share.access_type not in ['viewer', 'editor']:
        logger.warning(f"Invalid access type: {share.access_type}")
        raise HTTPException(status_code=400, detail="Type d'accès invalide")
    
    # Vérifier si le partage existe déjà
    existing_share = db.query(Share).filter(
        Share.upload_id == upload_id,
        Share.user_id == share.user_id
    ).first()
    
    if existing_share:
        existing_share.access_type = share.access_type  # Mettre à jour le type d'accès
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
    Supprimer le partage d'un enregistrement avec un utilisateur.
    """
    # Vérifier que l'enregistrement appartient à l'utilisateur actuel
    upload = db.query(Upload).filter(
        Upload.id == upload_id,
        Upload.owner_id == current_user.id
    ).first()
    
    if not upload:
        logger.warning(f"Upload not found or not owned by user {current_user.username}: {upload_id}")
        raise HTTPException(status_code=404, detail="Enregistrement non trouvé")
    
    # Trouver le partage
    share = db.query(Share).filter(
        Share.upload_id == upload_id,
        Share.user_id == user_id
    ).first()
    
    if not share:
        logger.warning(f"Share not found for user {user_id} on upload {upload_id}")
        raise HTTPException(status_code=404, detail="Partage non trouvé")
    
    db.delete(share)
    try:
        db.commit()
        logger.info(f"Removed share for user {user_id} on upload {upload_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Error removing share: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove share")
    
    return {"message": "Accès de partage supprimé avec succès"}

# Punctuation mapping and replacement function
PUNCTUATION_MAP = {
    "point": ".",
    "virgule": ",",
    "nouvelle ligne": "\n",
    "À la ligne": "\n",
    "a la ligne": "\n",
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
    "plus": "+",
    "moins": "-",
    "égal": "=",
    "inférieur": "<",
    "supérieur": ">",
    "crochet ouvrant": "[",
    "crochet fermant": "]",
    "accolade ouvrante": "{",
    "accolade fermante": "}",
    "entre parenthèses": "(",
    "Entre parenthèses": "(",
    "Fermez la parenthèse": ")",
    "fermez la parenthèse": ")",
    "nouvelle ligne": "\n",
    "à la ligne": "\n",
    "ouvrez la parenthèse" : "(",
    "fermez la parenthèse" : ")",
    "tiret du 6": "-",
    "à la ligne:":"\n ",
    "points de suspension": "...",
    "double point": "..",
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
    "guillemets anglais ouvrants": """,
    "guillemets anglais fermants": """,
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
    "exposant trois": "³"



}

def remplacer_ponctuation(transcription: str) -> str:
    """
    Replace punctuation words with corresponding punctuation marks in a case-insensitive manner.
    
    Args:
        transcription: Input text containing punctuation words
    Returns:
        str: Text with punctuation words replaced by actual punctuation marks
    """
    # Convert input to lowercase for case-insensitive comparison
    result = transcription
    
    # Create case-insensitive pattern replacements
    for mot, signe in PUNCTUATION_MAP.items():
        # Create patterns for different cases:
        # 1. All lowercase
        # 2. First letter capitalized
        # 3. All uppercase
        patterns = [
            mot.lower(),
            mot.capitalize(),
            mot.upper()
        ]
        
        # Apply each pattern
        for pattern in patterns:
            result = result.replace(pattern, signe)
    
    return result


