# models.py
import json
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    uploads = relationship("Upload", back_populates="owner")

class Upload(Base):
    __tablename__ = 'uploads'
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=False, index=True, nullable=False)
    transcription_filename = Column(String, unique=False, index=True, nullable=False)
    upload_time = Column(DateTime, default=datetime.datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.id'))
    is_archived = Column(Boolean, default=False)
    shared_with = Column(String, default='[]')  # Store as JSON string
    
    owner = relationship("User", back_populates="uploads")

    def get_shared_users(self) -> list[int]:
        """Get list of user IDs this upload is shared with"""
        try:
            if not self.shared_with:
                return []
            return json.loads(self.shared_with)
        except json.JSONDecodeError:
            return []

    def add_shared_user(self, user_id: int):
        """Add a user ID to shared_with list"""
        current_users = self.get_shared_users()
        if user_id not in current_users:
            current_users.append(user_id)
            self.shared_with = json.dumps(current_users)

    def remove_shared_user(self, user_id: int):
        """Remove a user ID from shared_with list"""
        current_users = self.get_shared_users()
        if user_id in current_users:
            current_users.remove(user_id)
            self.shared_with = json.dumps(current_users)