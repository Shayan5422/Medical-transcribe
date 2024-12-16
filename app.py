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

app = FastAPI()

# Initialiser la base de données
init_db()

# Dépendance pour obtenir la session de base de données
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Configuration CORS
origins = [
    "http://localhost:4200",  # Adresse de votre application Angular
    # Ajoutez d'autres origines si nécessaire
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration de l'appareil et du type de données
device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Identifiant du modèle
model_id = "openai/whisper-large-v3"

# Chargement du modèle et du processeur
print("Loading model and processor...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
print("Model and processor loaded.")

# Création du pipeline de reconnaissance vocale
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs={"language": "french"}
)

# Dossier pour stocker les fichiers audio
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Création d'un pool de threads pour les opérations bloquantes
executor = ThreadPoolExecutor(max_workers=4)  # Ajustez le nombre de threads selon vos besoins

@app.post("/upload-audio/")
async def upload_audio(
    user_id: int = Form(...), 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    """
    Endpoint pour uploader un fichier audio et obtenir sa transcription.
    """
    print(f"Received file: {file.filename}, content_type: {file.content_type}, from user_id: {user_id}")
    
    # Vérifier si l'utilisateur existe, sinon le créer
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        user = User(id=user_id, username=f"user_{user_id}")
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"Created new user with id: {user_id}")
    
    # Extraire le type principal du content_type
    content_type_main = file.content_type.split(';')[0].strip()
    
    # Liste des types audio supportés
    supported_types = [
        "audio/wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/x-wav",
        "audio/webm",
        "audio/ogg",
        "audio/flac",
    ]
    
    # Vérifier si le type est supporté
    if content_type_main not in supported_types:
        print("Unsupported file type.")
        raise HTTPException(status_code=400, detail="Format de fichier audio non supporté.")
    
    # Sauvegarder le fichier audio
    original_filename = file.filename
    file_path = os.path.join(AUDIO_DIR, original_filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    print(f"File saved to {file_path}")
    
    # Convertir l'audio en wav si nécessaire
    if content_type_main != "audio/wav":
        try:
            audio = AudioSegment.from_file(file_path)
            wav_filename = os.path.splitext(original_filename)[0] + ".wav"
            wav_path = os.path.join(AUDIO_DIR, wav_filename)
            audio = audio.set_channels(1)  # Convertir en mono
            audio.export(wav_path, format="wav")
            os.remove(file_path)  # Supprimer le fichier original
            file_path = wav_path
            original_filename = wav_filename  # Mettre à jour le nom du fichier
            print(f"Converted file to {wav_path}")
        except Exception as e:
            print(f"Error converting audio: {e}")
            raise HTTPException(status_code=400, detail=f"Erreur lors de la conversion de l'audio : {e}")
    
    # Charger et traiter l'audio de manière asynchrone
    try:
        transcription = await asyncio.get_event_loop().run_in_executor(executor, process_transcription, file_path)
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la transcription de l'audio : {e}")
    
    # Sauvegarder la transcription dans un fichier texte
    transcription_filename = f"{os.path.splitext(os.path.basename(original_filename))[0]}_transcription.txt"
    transcription_path = os.path.join(AUDIO_DIR, transcription_filename)
    with open(transcription_path, "w", encoding="utf-8") as f:
        f.write(transcription.strip())
    print(f"Transcription saved to {transcription_path}")
    
    # Enregistrer les informations dans la base de données
    upload_record = Upload(
        filename=original_filename,  # Utiliser le nom du fichier sauvegardé
        transcription_filename=transcription_filename,
        owner=user
    )
    db.add(upload_record)
    db.commit()
    db.refresh(upload_record)
    print(f"Upload record created with id: {upload_record.id}")
    
    return JSONResponse(content={
        "transcription": transcription.strip(),
        "transcription_file": transcription_filename,
        "upload_id": upload_record.id
    })

def process_transcription(file_path):
    """
    Fonction bloquante pour traiter la transcription de l'audio.
    """
    print(f"Processing transcription for {file_path}")
    # Charger l'audio avec SoundFile
    audio_data, samplerate = sf.read(file_path)
    print(f"Audio data loaded: {len(audio_data)} samples at {samplerate} Hz")
    
    # Resampler si nécessaire
    if samplerate != 16000:
        try:
            audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)
            samplerate = 16000
            print(f"Audio resampled to {samplerate} Hz")
        except Exception as e:
            print(f"Error resampling audio: {e}")
            raise e
    
    # Diviser l'audio en segments de 30 secondes
    segments = diviser_audio(audio_data, samplerate, duree_max=30)
    print(f"Audio divided into {len(segments)} segment(s).")
    
    transcription_complete = ""
    
    for idx, segment in enumerate(segments):
        try:
            print(f"Processing segment {idx + 1}/{len(segments)}")
            # Utiliser le pipeline ASR pour la transcription
            transcription = asr_pipeline(segment)["text"]
            transcription_complete += transcription + " "
            print(f"Segment {idx + 1} transcription: {transcription}")
        except Exception as e:
            print(f"Error processing segment {idx + 1}: {e}")
            raise e
    
    return transcription_complete

@app.get("/download-transcription/{upload_id}")
async def download_transcription(upload_id: int, db: Session = Depends(get_db)):
    """
    Endpoint pour télécharger la transcription d'un upload spécifique.
    """
    upload_record = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload_record:
        print(f"Transcription record not found for upload_id: {upload_id}")
        raise HTTPException(status_code=404, detail="Fichier de transcription non trouvé.")
    
    transcription_path = os.path.join(AUDIO_DIR, upload_record.transcription_filename)
    if not os.path.exists(transcription_path):
        print(f"Transcription file not found: {upload_record.transcription_filename}")
        raise HTTPException(status_code=404, detail="Fichier de transcription non trouvé.")
    
    print(f"Transcription file found: {upload_record.transcription_filename}")
    return FileResponse(transcription_path, media_type='text/plain', filename=upload_record.transcription_filename)

@app.get("/download-audio/{upload_id}")
async def download_audio(upload_id: int, db: Session = Depends(get_db)):
    """
    Endpoint pour télécharger le fichier audio original d'un upload spécifique.
    """
    upload_record = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload_record:
        print(f"Audio record not found for upload_id: {upload_id}")
        raise HTTPException(status_code=404, detail="Fichier audio non trouvé.")
    
    audio_path = os.path.join(AUDIO_DIR, upload_record.filename)
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {upload_record.filename}")
        raise HTTPException(status_code=404, detail="Fichier audio non trouvé.")
    
    print(f"Audio file found: {upload_record.filename}")
    return FileResponse(audio_path, media_type='audio/wav', filename=upload_record.filename)

@app.get("/history/{user_id}")
async def get_history(user_id: int, db: Session = Depends(get_db)):
    """
    Endpoint pour récupérer l'historique des uploads d'un utilisateur.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        print(f"User not found with id: {user_id}")
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé.")
    
    uploads = db.query(Upload).filter(Upload.user_id == user_id).all()
    history = []
    for upload in uploads:
        history.append({
            "upload_id": upload.id,
            "filename": upload.filename,
            "transcription_filename": upload.transcription_filename,
            "upload_time": upload.upload_time.isoformat()
        })
    
    print(f"Retrieved history for user_id: {user_id}, total uploads: {len(history)}")
    return JSONResponse(content={"history": history})

@app.get("/get-transcription/{upload_id}")
async def get_transcription(upload_id: int, db: Session = Depends(get_db)):
    """
    Endpoint pour récupérer le texte de la transcription d'un upload spécifique.
    """
    upload_record = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload_record:
        print(f"Transcription record not found for upload_id: {upload_id}")
        raise HTTPException(status_code=404, detail="Transcription non trouvée.")
    
    transcription_path = os.path.join(AUDIO_DIR, upload_record.transcription_filename)
    if not os.path.exists(transcription_path):
        print(f"Transcription file not found: {upload_record.transcription_filename}")
        raise HTTPException(status_code=404, detail="Fichier de transcription non trouvé.")
    
    with open(transcription_path, "r", encoding="utf-8") as f:
        transcription = f.read()
    
    return {"transcription": transcription}

def diviser_audio(audio, samplerate=16000, duree_max=30):
    """
    Divise l'audio en segments de durée maximale spécifiée.

    :param audio: Tableau numpy contenant les données audio
    :param samplerate: Taux d'échantillonnage (Hz)
    :param duree_max: Durée maximale de chaque segment en secondes
    :return: Liste de segments audio
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


