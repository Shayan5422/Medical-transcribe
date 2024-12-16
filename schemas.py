# schemas.py
from pydantic import BaseModel
from typing import Optional, List

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UploadHistory(BaseModel):
    upload_id: int
    filename: str
    transcription_filename: str
    upload_time: str

class UploadHistoryResponse(BaseModel):
    history: List[UploadHistory]
