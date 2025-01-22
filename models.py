# models.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship, declarative_base
import datetime
import pytz

Base = declarative_base()

def get_current_paris_time():
    """
    Returns current time in France timezone
    """
    return datetime.datetime.now(pytz.timezone('Europe/Paris'))

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    
    uploads = relationship("Upload", back_populates="owner", cascade="all, delete-orphan")
    shares = relationship("Share", back_populates="user", cascade="all, delete-orphan")

class Upload(Base):
    __tablename__ = 'uploads'
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    transcription_filename = Column(String, nullable=True)
    upload_time = Column(DateTime(timezone=True), default=get_current_paris_time)
    owner_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    is_archived = Column(Boolean, default=False)
    audio_deleted = Column(Boolean, default=False)  # New column to track if audio has been deleted
    
    owner = relationship("User", back_populates="uploads")
    shares = relationship("Share", back_populates="upload", cascade="all, delete-orphan")
    
    @property
    def should_delete_audio(self) -> bool:
        """Check if audio file should be deleted (7 days old)"""
        if self.audio_deleted:
            return False
            
        if not self.upload_time:
            return False
            
        current_time = get_current_paris_time()
        age = current_time - self.upload_time
        return age.days >= 7

    def has_access(self, user_id: int) -> bool:
        return user_id == self.owner_id or any(share.user_id == user_id for share in self.shares)
    
    def get_shared_users(self) -> list[int]:
        return [share.user_id for share in self.shares]
    
    def add_shared_user(self, user_id: int, access_type: str = 'viewer'):
        if not any(share.user_id == user_id for share in self.shares):
            new_share = Share(user_id=user_id, access_type=access_type)
            self.shares.append(new_share)
    
    def remove_shared_user(self, user_id: int):
        self.shares = [share for share in self.shares if share.user_id != user_id]

class Share(Base):
    __tablename__ = 'shares'
    
    id = Column(Integer, primary_key=True, index=True)
    upload_id = Column(Integer, ForeignKey('uploads.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    access_type = Column(String, nullable=False)
    
    upload = relationship("Upload", back_populates="shares")
    user = relationship("User", back_populates="shares")