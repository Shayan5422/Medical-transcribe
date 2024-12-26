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


class ShareInfo(BaseModel):
    user_id: int
    access_type: str
class TokenData(BaseModel):
    username: Optional[str] = None
class UploadHistory(BaseModel):
    upload_id: int
    filename: str
    transcription_filename: str
    upload_time: str
    is_archived: bool = False
    shared_with: List[ShareInfo] = []
    owner_id: int

    class Config:
        orm_mode = True

class UploadHistoryResponse(BaseModel):
    history: List[UploadHistory]
    current_user_id: int  # Add this line


class ShareCreate(BaseModel):
    user_id: int
    access_type: str  # 'viewer' ou 'editor'

class ShareResponse(BaseModel):
    id: int
    upload_id: int
    user_id: int
    access_type: str

    class Config:
        orm_mode = True



