import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Integer, Float, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

# User models
class User(Base):
    __tablename__ = "User"
    
    id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=True)
    email = Column(String(255), unique=True, nullable=True)
    emailVerified = Column(DateTime, nullable=True)
    image = Column(String(255), nullable=True)
    
    # Relationships
    accounts = relationship("Account", back_populates="user", cascade="all, delete")
    sessions = relationship("Session", back_populates="user", cascade="all, delete")

class Account(Base):
    __tablename__ = "Account"
    
    id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    userId = Column(String(255), ForeignKey("User.id", ondelete="CASCADE"))
    type = Column(String(255))
    provider = Column(String(255))
    providerAccountId = Column(String(255))
    refresh_token = Column(Text, nullable=True)
    access_token = Column(Text, nullable=True)
    expires_at = Column(Integer, nullable=True)
    token_type = Column(String(255), nullable=True)
    scope = Column(String(255), nullable=True)
    id_token = Column(Text, nullable=True)
    session_state = Column(String(255), nullable=True)
    
    # Relationship
    user = relationship("User", back_populates="accounts")
    
    # Unique constraint
    __table_args__ = (
        {"mysql_collate": "utf8mb4_unicode_ci"},
    )

class Session(Base):
    __tablename__ = "Session"
    
    id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    sessionToken = Column(String(255), unique=True)
    userId = Column(String(255), ForeignKey("User.id", ondelete="CASCADE"))
    expires = Column(DateTime)
    
    # Relationship
    user = relationship("User", back_populates="sessions")

class VerificationToken(Base):
    __tablename__ = "VerificationToken"
    
    identifier = Column(String(255), primary_key=True)
    token = Column(String(255), unique=True, primary_key=True)
    expires = Column(DateTime)

# Music models
class Song(Base):
    __tablename__ = "songs"
    
    song_id = Column(String(36), primary_key=True)
    track_name = Column(String(255), nullable=True)
    track_artist = Column(String(255), nullable=True)
    track_album_name = Column(String(255), nullable=True)
    original_lyrics = Column(Text, nullable=True)
    cleaned_lyrics = Column(Text, nullable=True)
    playlist_genre = Column(String(100), nullable=True, index=True)
    playlist_subgenre = Column(String(100), nullable=True)
    language = Column(String(50), nullable=True, index=True)
    sentiment = Column(String(50), nullable=True)
    popularity_score = Column(Float, nullable=True)
    link = Column(String(1024), nullable=True)
    spotify_id = Column(String(64), nullable=True, index=True)
    siamzone_id = Column(String(64), nullable=True, index=True)
    
    # Relationships
    features = relationship("SongFeature", back_populates="song", uselist=False)
    embedding = relationship("SongEmbeddingVector", back_populates="song", uselist=False)

class SongFeature(Base):
    __tablename__ = "song_features"
    
    song_id = Column(String(36), ForeignKey("songs.song_id", ondelete="NO ACTION", onupdate="NO ACTION"), primary_key=True)
    danceability = Column(Float, nullable=True)
    energy = Column(Float, nullable=True)
    key = Column(Float, nullable=True)
    loudness = Column(Float, nullable=True)
    mode = Column(Float, nullable=True)
    speechiness = Column(Float, nullable=True)
    acousticness = Column(Float, nullable=True)
    instrumentalness = Column(Float, nullable=True)
    liveness = Column(Float, nullable=True)
    valence = Column(Float, nullable=True)
    tempo = Column(Float, nullable=True)
    
    # Relationship
    song = relationship("Song", back_populates="features")

# Note: This model contains a vector type which is not directly supported by SQLAlchemy
# You'll need to use a custom dialect or handle this specifically for your database
class SongEmbeddingVector(Base):
    __tablename__ = "song_embeddings_vector"
    
    song_id = Column(String(36), ForeignKey("songs.song_id", ondelete="NO ACTION", onupdate="NO ACTION"), primary_key=True)
    # For vector fields, we'll use Text as a placeholder
    # You'll need custom handling for actual vector operations
    lyrics_embedding = Column(Text, nullable=True)  # This should be a vector(4096)
    metadata_embedding = Column(Text, nullable=True)  # This should be a vector(4096)
    
    # Relationship
    song = relationship("Song", back_populates="embedding")

class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), index=True, nullable=False)
    session_id = Column(String(255), index=True, nullable=False)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    mode = Column(String(50), nullable=True)
    intent = Column(String(100), nullable=True)  
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):  
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "query": self.query,
            "response": self.response,
            "mode": self.mode,
            "intent": self.intent,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }