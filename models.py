# models.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship, declarative_base
import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    
    # Relation avec Upload
    uploads = relationship("Upload", back_populates="owner", cascade="all, delete-orphan")
    
    # Relation avec Share
    shares = relationship("Share", back_populates="user", cascade="all, delete-orphan")


class Upload(Base):
    __tablename__ = 'uploads'
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    transcription_filename = Column(String, nullable=True)
    upload_time = Column(DateTime, default=datetime.datetime.utcnow)
    owner_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    is_archived = Column(Boolean, default=False)
    
    # Relation avec User
    owner = relationship("User", back_populates="uploads")
    
    # Relation avec Share
    shares = relationship("Share", back_populates="upload", cascade="all, delete-orphan")
    
    def has_access(self, user_id: int) -> bool:
        """Vérifie si un utilisateur a accès à cet upload"""
        return user_id == self.owner_id or any(share.user_id == user_id for share in self.shares)
    
    def get_shared_users(self) -> list[int]:
        """Retourne la liste des IDs des utilisateurs avec qui cet upload est partagé"""
        return [share.user_id for share in self.shares]
    
    def add_shared_user(self, user_id: int, access_type: str = 'viewer'):
        """Ajoute un partage avec un utilisateur"""
        if not any(share.user_id == user_id for share in self.shares):
            new_share = Share(user_id=user_id, access_type=access_type)
            self.shares.append(new_share)
    
    def remove_shared_user(self, user_id: int):
        """Supprime un partage avec un utilisateur"""
        self.shares = [share for share in self.shares if share.user_id != user_id]


class Share(Base):
    __tablename__ = 'shares'
    
    id = Column(Integer, primary_key=True, index=True)
    upload_id = Column(Integer, ForeignKey('uploads.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    access_type = Column(String, nullable=False)  # 'viewer' ou 'editor'
    
    # Relations
    upload = relationship("Upload", back_populates="shares")
    user = relationship("User", back_populates="shares")
