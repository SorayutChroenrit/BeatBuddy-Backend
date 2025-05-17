from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime

# User schemas
class UserBase(BaseModel):
    email: Optional[str]
    name: Optional[str]
    image: Optional[str]

class User(UserBase):
    id: str
    
    class Config:
        from_attributes = True

# Music schemas
class ChatMessage(BaseModel):
    user_id: str
    session_id: str
    query: str
    response: str
    mode: Optional[str] = "buddy"

class AskQuestionRequest(BaseModel):
    question: str
    mode: Optional[str] = "buddy"
    session_id: Optional[str] = "default"

class AskQuestionResponse(BaseModel):
    response: str
    mode: str
    intent: str
    sources: List[Dict[str, Any]]
    processing_time: float

class SongInfo(BaseModel):
    id: str
    song_title: str
    artist: str
    processed_text: Optional[str] = None
    spotify_id: Optional[str] = None
    popularity_score: Optional[float] = None
    similarity: float = 0.0
    match_type: Optional[str] = None
    has_duplicates: bool = False

# Session schema
class SessionData(BaseModel):
    user: Optional[User] = None