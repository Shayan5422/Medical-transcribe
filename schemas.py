# schemas.py
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

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
    is_archived: bool = False
    shared_with: List[int] = []

    class Config:
        from_attributes = True

class UploadHistoryResponse(BaseModel):
    history: List[UploadHistory]

class ShareResponse(BaseModel):
    share_link: str