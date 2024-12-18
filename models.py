# models.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)  # Password storage
    uploads = relationship("Upload", back_populates="owner")

class Upload(Base):
    __tablename__ = 'uploads'
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=False, index=True, nullable=False)
    transcription_filename = Column(String, unique=False, index=True, nullable=False)
    upload_time = Column(DateTime, default=datetime.datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.id'))
    
    owner = relationship("User", back_populates="uploads")
