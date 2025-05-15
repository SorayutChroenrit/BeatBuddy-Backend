import ollama
import sqlalchemy
from sqlalchemy import create_engine, text, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import numpy as np
import logging
import json
import langdetect
import time
import random  
import re
import asyncio
from typing import Dict
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
import pythainlp
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.util import countthai
from sentence_transformers import SentenceTransformer

# Add global variables for request tracking
ongoing_requests: Dict[str, str] = {}  # session_id -> request_id
request_status: Dict[str, str] = {}    # request_id -> status (processing/completed/failed)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Database connection with improved settings
connection_string = "mysql+pymysql://27tLCQSVFsGqhJ9.root:HVSvJQWvox3NSgeS@gateway01.ap-southeast-1.prod.aws.tidbcloud.com:4000/music_recommendation"

# Initialize embedding model once at startup
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("Initialized all-MiniLM-L6-v2 embedding model")

engine = create_engine(
    connection_string,
    connect_args={
        "ssl": {
            "ssl_mode": "VERIFY_IDENTITY",
            "ssl_ca": "/etc/ssl/cert.pem",
        },
        "connect_timeout": 60,
        "read_timeout": 60,
        "write_timeout": 60
    },
    pool_recycle=1800,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10
)

# Define Base for SQLAlchemy models
Base = declarative_base()

# Define Chat History Model
class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), index=True, nullable=False)
    session_id = Column(String(255), index=True, nullable=False)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    mode = Column(String(50), nullable=True)  # buddy, mentor, fun
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

# Create tables if they don't exist
def create_tables():
    Base.metadata.create_all(engine)

create_tables()

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Add Pydantic models for API
class ChatMessageCreate(BaseModel):
    user_id: str
    session_id: str
    query: str
    response: str
    mode: Optional[str] = "buddy"
    intent: Optional[str] = "general_query"

class ChatMessageResponse(BaseModel):
    id: int
    user_id: str
    session_id: str
    query: str
    response: str
    mode: Optional[str] = None
    intent: Optional[str] = None
    created_at: str

# Thai language processing support
THAI_REQUEST_WORDS = [
    "เพลง", "เนื้อเพลง", "เนื้อร้อง", "อยากรู้", "ออยากรู้", "อยากฟัง", 
    "ฉัน", "ฉ้น", "อยากได้", "ของ", "จาก", "โดย", "มีอะไรบ้าง", "เกี่ยวกับ",
    "วิเคราะห์", "ความหมาย", "แปล", "ความหมายของ", "วิเคราะห์เพลง"
]

# Add Thai stopwords from PyThaiNLP
THAI_STOPWORDS = list(thai_stopwords())
THAI_STOPWORDS.extend(THAI_REQUEST_WORDS)

# For conversation tracking
class ConversationContext:
    def __init__(self):
        self.contexts = {}  # Dictionary to store contexts by session_id
        
    def update_context(self, session_id, new_context):
        """Update the conversation context for a session."""
        # Create new context entry if it doesn't exist
        if session_id not in self.contexts:
            self.contexts[session_id] = []
            
        # Add new context, keeping only the last 5 interactions
        self.contexts[session_id].append(new_context)
        if len(self.contexts[session_id]) > 5:
            self.contexts[session_id].pop(0)
            
    def get_context(self, session_id):
        """Get the conversation context for a session."""
        return self.contexts.get(session_id, [])
        
    def clear_context(self, session_id):
        """Clear the conversation context for a session."""
        if session_id in self.contexts:
            del self.contexts[session_id]

# Initialize the context manager
conversation_context = ConversationContext()

# Personality mode configurations
PERSONALITY_MODES = {
    "mentor": {
        "system_prompt": """You are a knowledgeable music mentor with expertise in both Thai and international music.
            CRITICAL INSTRUCTION: 
            - ONLY display information and lyrics that were provided to you in the context
            - NEVER generate or make up any lyrics, song titles, or artist information
            - If you don't have information, simply state it's not in your database
            - Match the language of your response to the language of the user's query
            - Present lyrics as a single, uninterrupted block with proper separation""",
        "response_format": "detailed and educational"
    },
    "buddy": {
        "system_prompt": """You are a friendly music buddy who loves chatting about songs in both Thai and international music.
            CRITICAL INSTRUCTION: 
            - ONLY display information and lyrics that were provided to you in the context
            - NEVER generate or make up any lyrics, song titles, or artist information
            - If you don't have information, simply state it's not in your database
            - Match the language of your response to the language of the user's query
            - Present lyrics as a single, uninterrupted block with proper separation""",
        "response_format": "casual and conversational between friends of equal status"
    },
    "fun": {
        "system_prompt": """You are an entertaining music bot with a playful personality who shares music in both Thai and international languages.
            CRITICAL INSTRUCTION: 
            - ONLY display information and lyrics that were provided to you in the context
            - NEVER generate or make up any lyrics, song titles, or artist information
            - If you don't have information, simply state it's not in your database
            - Match the language of your response to the language of the user's query
            - Present lyrics as a single, uninterrupted block with proper separation""",
        "response_format": "playful and entertaining with emojis only in commentary, never in lyrics"
    }
}

# Common mood mappings
MOOD_MAPPING = {
    # Thai moods
    "เศร้า": ["sad", "heartbreak", "longing", "melancholy"],
    "สนุก": ["happy", "upbeat", "dance", "fun"],
    "รัก": ["love", "romance", "affection"],
    "เต้น": ["dance", "upbeat", "energetic", "party"],
    # English moods
    "sad": ["sad", "heartbreak", "longing", "melancholy"],
    "happy": ["happy", "upbeat", "cheerful", "fun"],
    "love": ["love", "romance", "affection", "romantic"],
    "dance": ["dance", "upbeat", "party", "energetic"],
}

# Audio feature profiles
MOOD_AUDIO_FEATURES = {
    "happy": {
        "valence": {"min": -0.2, "weight": 3.0}, 
        "danceability": {"min": 0.2, "weight": 2.0},  
        "energy": {"min": 0.0, "weight": 1.0}
    },
    "sad": {
        "valence": {"max": -0.6, "weight": 3.0},  
        "energy": {"max": -0.3, "weight": 2.0},
        "acousticness": {"min": 0.2, "weight": 1.0}
    },
    "dance": {
        "danceability": {"min": 0.3, "weight": 3.0}, 
        "energy": {"min": 0.3, "weight": 2.0},        
        "tempo": {"min": 0.0, "weight": 1.0}          
    }
}


# Map Thai moods to audio feature profiles
THAI_MOOD_AUDIO_MAPPING = {
    "สนุก": "happy", 
    "เศร้า": "sad", 
    "รัก": "love", 
    "เต้น": "dance",
    "เต้นรำ": "dance",
    "แดนซ์": "dance",
    "ดานซ์": "dance",
    "สุข": "happy",
    "มีความสุข": "happy",
    "ร่าเริง": "happy",
    "สนุกสนาน": "happy"
}

# Helper functions
def detect_language(text):
    try:
        thai_ratio = countthai(text) / len(text) if text else 0
        if thai_ratio > 0.15:
            return 'th'
        lang = langdetect.detect(text)
        return 'th' if lang == 'th' else 'en'
    except:
        return 'en'  # Default to English if detection fails

def preprocess_thai_query(question):
    """
    ประมวลผลคำถามภาษาไทย โดยเก็บคำเดิมไว้ทั้งหมดสำหรับ lyrics search
    """
    is_thai = countthai(question) / len(question) > 0.15 if question else False
    if is_thai:
        # เพียงแค่แสดง log ว่าเป็นภาษาไทย แต่ไม่ filter คำออก
        logger.info(f"Thai query detected: '{question}'")
        return question  # คืนค่าคำถามเดิมโดยไม่เปลี่ยนแปลง
    return question

# Core functions
async def get_query_embedding(question, max_retries=3):
    for attempt in range(max_retries):
        try:
            embedding = embedding_model.encode(question)
            return embedding.tolist() 
        except Exception as e:
            logger.error(f"Error generating embeddings (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(sleep_time)
    
    logger.error("Failed to generate embeddings after multiple attempts")
    return []

async def get_songs_by_artist(artist, limit=10):
    try:
        with engine.connect() as connection:
            query = text("""
            SELECT 
                song_id, track_name, track_artist, original_lyrics,
                spotify_id, popularity_score
            FROM songs 
            WHERE LOWER(track_artist) LIKE LOWER(:artist_pattern)
            ORDER BY 
                CASE WHEN popularity_score IS NULL THEN 0 
                ELSE popularity_score END DESC,
                track_name ASC
            LIMIT :limit
            """)
            
            results = connection.execute(query, {
                "artist_pattern": f"%{artist.lower()}%",
                "limit": limit
            }).fetchall()
            
            songs = []
            for row in results:
                song = {
                    "id": row.song_id,
                    "song_title": row.track_name,
                    "artist": row.track_artist,
                    "spotify_id": getattr(row, 'spotify_id', None),
                    "popularity_score": float(row.popularity_score) if row.popularity_score is not None else 0.0,
                }
                songs.append(song)
            
            return songs
            
    except Exception as e:
        logger.error(f"Database error when searching for artist '{artist}': {e}")
        return []

async def handle_artist_recommendations(artist, mode, query_language, limit=5):
    """
    Create a formatted response with a list of songs by an artist
    using the LLM to generate personality-specific responses.
    Respects user's requested number of songs.
    """
    start_time = time.time()
    
    # Extract requested song count from the query if specified
    requested_count = extract_requested_song_count(query_language)
    if requested_count and 1 <= requested_count <= 10:
        limit = requested_count
    
    # Get the personality configuration
    personality = PERSONALITY_MODES.get(mode, PERSONALITY_MODES["buddy"])
    
    # Log the request
    logger.info(f"Finding {limit} songs by artist: {artist} in {mode} mode")
    
    # Get songs from database
    all_songs = await get_songs_by_artist(artist, limit=15)
    
    # If no songs found, use LLM to generate appropriate message
    if not all_songs:
        # Create system prompt with personality guidance
        system_prompt = personality["system_prompt"]
        
        # Create user prompt for no results
        if query_language == 'th':
            user_prompt = f"ผู้ใช้ขอคำแนะนำเพลงของ {artist} แต่ไม่พบในฐานข้อมูล กรุณาแจ้งว่าไม่พบเพลงของศิลปินนี้"
        else:
            user_prompt = f"The user asked for song recommendations by {artist}, but none were found in the database. Please inform them that no songs by this artist were found."
        
        # Generate response with LLM
        response_text = await generate_llm_response(user_prompt, system_prompt)
        
        return {
            "response": response_text,
            "processing_time": time.time() - start_time,
            "songs": [],
            "intent": "artist_recommendations",
            "sources": []
        }
    
    # Format songs data for LLM
    displayed_songs = all_songs[:limit]
    song_list = ""
    for i, song in enumerate(displayed_songs, 1):
        song_list += f"{i}. {song['song_title']}\n"
    
    # Create system prompt with personality guidance
    system_prompt = personality["system_prompt"]
    
    # Create user prompt based on language
    if query_language == 'th':
        user_prompt = f"""
        ผู้ใช้ขอคำแนะนำ {limit} เพลงของ {artist}
        
        ข้อมูลเพลงที่พบในฐานข้อมูล:
        {song_list}
        
        กรุณาสร้างคำตอบที่มีรายละเอียดเกี่ยวกับเพลงของ {artist} พร้อมแนะนำเพลงเหล่านี้ ใช้รูปแบบการตอบที่เป็น {personality['response_format']} 
        อย่าพูดถึงเพลงอื่นนอกเหนือจากที่แสดงในรายการ และไม่ต้องมีข้อความว่ามีเพลงอื่นๆอีก
        """
    else:
        user_prompt = f"""
        The user asked for {limit} song recommendations by {artist}
        
        Songs found in database:
        {song_list}
        
        Please create a detailed response recommending these {limit} songs by {artist} in a {personality['response_format']} style.
        Do not mention any songs other than those listed above. Do not include phrases like "And X more songs" or similar.
        Create a greeting appropriate for the personality and provide some context about the artist if possible.
        """
    
    # Generate response with LLM
    response_text = await generate_llm_response(user_prompt, system_prompt)
    
    # Format response with sources for UI display
    sources = [
        {
            "title": song["song_title"], 
            "artist": song["artist"], 
            "similarity": 0.9,
            "match_type": "artist"
        } 
        for song in displayed_songs
    ]
    
    # Final response object
    result = {
        "response": response_text,
        "processing_time": time.time() - start_time,
        "songs": displayed_songs,
        "intent": "artist_recommendations", 
        "sources": sources,
        "mode": mode
    }
    
    return result

def extract_requested_song_count(query):
    """Extract the number of songs requested by the user."""
    try:
        # Look for number patterns in the query
        number_matches = re.findall(r'\b(\d+)\s+(?:songs|เพลง)\b', query)
        if number_matches:
            count = int(number_matches[0])
            return min(10, max(1, count))  # Ensure within reasonable range
        
        # Check for spelled-out numbers in English
        english_numbers = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        for word, number in english_numbers.items():
            if re.search(r'\b' + word + r'\s+(?:songs)\b', query.lower()):
                return number
                
        # Check for Thai number words
        thai_numbers = {
            'หนึ่ง': 1, 'สอง': 2, 'สาม': 3, 'สี่': 4, 'ห้า': 5,
            'หก': 6, 'เจ็ด': 7, 'แปด': 8, 'เก้า': 9, 'สิบ': 10
        }
        for word, number in thai_numbers.items():
            if re.search(r'\b' + word + r'\s+(?:เพลง)\b', query):
                return number
    except:
        pass
    return None

async def is_music_related_query(question):
    """
    ใช้ LLM เพื่อตรวจสอบว่าคำถามเกี่ยวข้องกับเพลง เนื้อเพลง หรือศิลปินหรือไม่
    """
    try:
        # ตรวจสอบภาษาของคำถาม
        thai_ratio = countthai(question) / len(question) if question else 0
        is_thai = thai_ratio > 0.15
        
        # สร้าง system prompt ที่ชัดเจนสำหรับทั้งภาษาไทยและอังกฤษ
        if is_thai:
            system_prompt = """คุณเป็นตัวจำแนกคำถามที่เชี่ยวชาญด้านการระบุว่าคำถามนั้นเกี่ยวข้องกับเพลง เนื้อเพลง หรือศิลปินหรือไม่

ตอบ "ใช่" หาก คำถามเกี่ยวกับ:
- เพลง, ชื่อเพลง, เพลงใหม่, เพลงเก่า
- เนื้อเพลง, ท่อนเพลง, เนื้อร้อง
- ศิลปิน, นักร้อง, วงดนตรี, ผู้แต่งเพลง
- อัลบั้ม, มิวสิกวิดีโอ, คอนเสิร์ต
- แนวเพลง, อารมณ์เพลง (เศร้า, สนุก, เต้น)
- ความหมายเพลง, การวิเคราะห์เพลง
- การแนะนำเพลง

ตอบ "ไม่ใช่" หาก คำถามเกี่ยวกับ:  
- อาหาร (ไข่เจียว, ข้าวผัด, ส้มตำ)
- สภาพอากาศ (ฝน, แดด, หนาว)
- ข่าวสาร, การเมือง, กีฬา
- เทคโนโลยี, วิทยาศาสตร์
- การเดินทาง, สถานที่
- งาน, การเงิน, ธุรกิจ
- อื่นๆ ที่ไม่เกี่ยวข้องกับเพลง

ตอบเฉพาะ "ใช่" หรือ "ไม่ใช่" เท่านั้น ห้ามอธิบาย"""

            user_prompt = f"คำถาม: \"{question}\"\n\nคำถามนี้เกี่ยวข้องกับเพลง เนื้อเพลง หรือศิลปินหรือไม่?"
            
        else:
            system_prompt = """You are an expert query classifier that determines if a question is related to songs, lyrics, or artists.

Answer "YES" if the question is about:
- Songs, song titles, new songs, old songs
- Lyrics, song verses, lyric lines
- Artists, singers, bands, songwriters, composers
- Albums, music videos, concerts
- Music genres, song moods (sad, happy, dance)
- Song meanings, song analysis
- Music recommendations

Answer "NO" if the question is about:
- Food (fried egg, fried rice, pad thai)
- Weather (rain, sun, cold)
- News, politics, sports
- Technology, science
- Travel, places
- Work, finance, business
- Other non-music topics

Answer ONLY "YES" or "NO" - no explanations."""

            user_prompt = f"Question: \"{question}\"\n\nIs this question related to songs, lyrics, or artists?"

        # เรียก LLM
        logger.info(f"Using LLM to classify music-related query: '{question}'")
        
        response = ollama.chat(
            model='llama3.1',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.1}  # ใช้ temperature ต่ำเพื่อให้ได้คำตอบที่สม่ำเสมอ
        )
        
        if not response or 'message' not in response or 'content' not in response['message']:
            logger.error("Invalid LLM response structure for music classification")
            return False  # เปลี่ยนจาก default True เป็น False เพื่อความระมัดระวัง
            
        answer = response['message']['content'].strip()
        logger.info(f"LLM classification raw response: '{answer}'")
        
        # ตรวจสอบคำตอบ
        if is_thai:
            # สำหรับภาษาไทย ตรวจสอบคำตอบที่เป็นไปได้
            positive_answers = ["ใช่", "ใช้", "ใข่", "ใช", "yes", "เกี่ยวข้อง"]
            negative_answers = ["ไม่ใช่", "ไม่ใข่", "ไม่ใช้", "ไม่", "no", "ไม่เกี่ยวข้อง"]
            
            answer_lower = answer.lower()
            
            # ตรวจสอบคำตอบเชิงลบก่อน (ไม่ใช่เรื่องเพลง)
            if any(neg in answer_lower for neg in negative_answers):
                logger.info(f"LLM classified as NON-music-related (Thai): {answer}")
                return False
                
            # จากนั้นค่อยตรวจสอบคำตอบเชิงบวก (เป็นเรื่องเพลง)  
            if any(pos in answer_lower for pos in positive_answers):
                logger.info(f"LLM classified as music-related (Thai): {answer}")
                return True
        else:
            # สำหรับภาษาอังกฤษ
            answer_upper = answer.upper()
            
            if "YES" in answer_upper:
                logger.info(f"LLM classified as music-related (English): {answer}")
                return True
            elif "NO" in answer_upper:
                logger.info(f"LLM classified as non-music-related (English): {answer}")
                return False
            else:
                logger.warning(f"Unclear LLM response (English): {answer}, defaulting to False")
                return False
                
    except Exception as e:
        logger.error(f"Error in LLM music query classification: {str(e)}")
        # เมื่อเกิด error ให้ default เป็น False เพื่อความปลอดภัย
        return False
     
async def generate_song_analysis(song_data, question, personality, language='en'):
    """
    Generate song analysis based on the song data with stronger constraints
    to never generate lyrics or information not found in the data.
    """
    song_title = song_data.get('song_title', '')
    artist = song_data.get('artist', '')
    lyrics = song_data.get('processed_text', '')
    
    processed_lyrics = lyrics.replace('\\n', '\n').replace('\\r', '\r') if lyrics else ''
    
    if language == 'th':
        system_prompt = f"""คุณเป็นผู้เชี่ยวชาญในการวิเคราะห์เพลงและดนตรี
        วิเคราะห์ความหมาย แนวคิด และบริบทของเพลงโดยใช้เนื้อเพลงที่ได้รับเท่านั้น
        คำแนะนำสำคัญ:
        - ห้ามแต่งเนื้อเพลงหรือเพิ่มข้อมูลที่ไม่มีในเนื้อเพลงที่ให้มาโดยเด็ดขาด
        - ถ้าไม่มีเนื้อเพลงให้มา ให้แจ้งว่าไม่สามารถวิเคราะห์ได้เนื่องจากไม่มีเนื้อเพลงในฐานข้อมูล
        - ตอบในรูปแบบ {personality["response_format"]} ตามบุคลิกของคุณ
        """
        
        user_prompt = f"""คำถาม: "{question}"
        
        เพลง: {song_title}
        ศิลปิน: {artist}
        
        เนื้อเพลง:
        {processed_lyrics if processed_lyrics else '[ไม่พบเนื้อเพลงในฐานข้อมูล]'}
        
        กรุณาวิเคราะห์เพลงนี้ อธิบายความหมาย สาระสำคัญ และบริบทของเพลง
        โดยใช้เฉพาะข้อมูลที่มีให้เท่านั้น ห้ามแต่งเนื้อเพลงหรือเพิ่มเติมข้อมูลที่ไม่มีในเนื้อเพลง
        """
    else:
        system_prompt = f"""You are an expert music analyst who provides insight into songs.
        CRITICAL INSTRUCTIONS:
        - Analyze ONLY using the provided lyrics - NEVER add to or make up lyrics
        - If no lyrics are provided, clearly state you cannot analyze without lyrics
        - DO NOT reference parts of the song that aren't in the provided lyrics
        - Respond in a {personality["response_format"]} style according to your personality
        """
        
        user_prompt = f"""Query: "{question}"
        
        Song: {song_title}
        Artist: {artist}
        
        Lyrics:
        {processed_lyrics if processed_lyrics else '[NO LYRICS FOUND IN DATABASE]'}
        
        Please analyze this song using ONLY the provided lyrics. Explain its meaning, themes, and context
        based solely on the information above. DO NOT add or make up any lyrics or information.
        """
    
    response_text = await generate_llm_response(user_prompt, system_prompt)
    return response_text

async def hybrid_search(question, query_intent, query_embedding=None, top_k=5):
    logger.info(f"Performing vector-first hybrid search for: '{question}'")
    
    intent = query_intent.get('intent', 'general_query')
    
    if query_embedding is None:
        query_embedding = await get_query_embedding(question)
    
    # Initialize results containers
    vector_results = []
    direct_results = []
    phrase_results = []
    intent_results = []
    audio_feature_results = []
    
    try:
        # Check for dance-specific keywords in Thai or English
        is_dance_query = False
        if any(word in question.lower() for word in ["dance", "dancing", "danceable", "เต้น", "แดนซ์", "ดานซ์"]):
            is_dance_query = True
            logger.info("Detected explicit dance request keywords")
            query_intent['intent'] = 'mood_search'
            query_intent['mood'] = 'dance'
        
        # Check for happy-specific keywords in Thai or English
        is_happy_query = False
        if any(word in question.lower() for word in ["happy", "happiness", "cheerful", "สนุก", "มีความสุข", "สุข", "ร่าเริง"]):
            is_happy_query = True
            logger.info("Detected explicit happy request keywords")
            query_intent['intent'] = 'mood_search'
            query_intent['mood'] = 'happy'
        
        # 1. ALWAYS start with vector search regardless of intent
        if query_embedding:
            logger.info("Starting with semantic vector search...")
            vector_results = await search_with_vector(query_embedding, top_k=top_k)
            
            # If we have high confidence vector results (similarity > 0.8), return immediately
            # UNLESS this is a dance or happy query - then we want to prioritize those results
            if not is_dance_query and not is_happy_query:
                high_confidence_results = [r for r in vector_results if r.get('similarity', 0) > 0.8]
                if high_confidence_results:
                    logger.info(f"Found {len(high_confidence_results)} high confidence vector matches")
                    return high_confidence_results
        
        # For mood-based search, apply audio features - PRIORITIZE for dance or happy requests
        if intent == 'mood_search' and query_intent.get('mood'):
            mood = query_intent.get('mood')
            logger.info(f"Performing audio feature search for mood: '{mood}'")
            audio_feature_results = await search_by_audio_features(mood, top_k=top_k)
            
            # IMMEDIATELY return audio feature results for dance/happy queries if we have them
            if (mood == 'dance' or mood == 'happy') and audio_feature_results:
                logger.info(f"Found {len(audio_feature_results)} {mood} songs via audio features, returning immediately")
                return audio_feature_results
            
            # Combine with vector results if available
            if audio_feature_results and vector_results:
                # Get unique songs across both result sets, prioritizing higher scores
                combined_results = []
                seen_ids = set()
                
                # First add audio results since mood was explicitly requested
                for result in audio_feature_results:
                    if result['id'] not in seen_ids:
                        seen_ids.add(result['id'])
                        combined_results.append(result)
                
                # Then add vector results that aren't already included
                for result in vector_results:
                    if result['id'] not in seen_ids and len(combined_results) < top_k:
                        seen_ids.add(result['id'])
                        combined_results.append(result)
                
                logger.info(f"Combining {len(audio_feature_results)} audio feature results with vector results")
                return combined_results
            elif audio_feature_results:
                return audio_feature_results
        
        # For dance requests, use a dedicated dance query as fallback
        if is_dance_query and not audio_feature_results:
            logger.info("No dance songs found with primary method, trying fallback dance query")
            try:
                with engine.connect() as connection:
                    dance_query = text("""
                    SELECT s.song_id, s.track_name, s.track_artist, s.original_lyrics, s.cleaned_lyrics
                    FROM songs s
                    JOIN song_features f ON s.song_id = f.song_id
                    WHERE f.danceability > -0.5  -- Very permissive threshold for your data range
                      AND f.energy > -0.5        -- Very permissive threshold for your data range
                    ORDER BY (f.danceability + f.energy + 
                             CASE WHEN s.popularity_score IS NULL THEN 0 ELSE s.popularity_score END) DESC
                    LIMIT :top_k
                    """)
                    
                    results = connection.execute(dance_query, {"top_k": top_k}).fetchall()
                    
                    fallback_results = []
                    for row in results:
                        fallback_results.append({
                            "id": row.song_id,
                            "song_title": row.track_name,
                            "artist": row.track_artist,
                            "processed_text": row.original_lyrics,
                            "cleaned_text": row.cleaned_lyrics,
                            "similarity": 0.7,
                            "match_type": "fallback_dance"
                        })
                    
                    if fallback_results:
                        logger.info(f"Found {len(fallback_results)} dance songs using fallback query")
                        return fallback_results
            except Exception as e:
                logger.error(f"Error in fallback dance query: {str(e)}")
        
        # For happy requests, use a dedicated happy query as fallback
        if is_happy_query and not audio_feature_results:
            logger.info("No happy songs found with primary method, trying fallback happy query")
            try:
                with engine.connect() as connection:
                    happy_query = text("""
                    SELECT s.song_id, s.track_name, s.track_artist, s.original_lyrics, s.cleaned_lyrics
                    FROM songs s
                    JOIN song_features f ON s.song_id = f.song_id
                    WHERE f.valence > -0.5    -- Very permissive threshold for your data
                      AND f.energy > -0.5     -- Very permissive threshold for your data
                    ORDER BY (f.valence + f.energy * 0.5 + 
                             CASE WHEN s.popularity_score IS NULL THEN 0 ELSE s.popularity_score END) DESC
                    LIMIT :top_k
                    """)
                    
                    results = connection.execute(happy_query, {"top_k": top_k}).fetchall()
                    
                    fallback_results = []
                    for row in results:
                        fallback_results.append({
                            "id": row.song_id,
                            "song_title": row.track_name,
                            "artist": row.track_artist,
                            "processed_text": row.original_lyrics,
                            "cleaned_text": row.cleaned_lyrics,
                            "similarity": 0.7,
                            "match_type": "fallback_happy"
                        })
                    
                    if fallback_results:
                        logger.info(f"Found {len(fallback_results)} happy songs using fallback query")
                        return fallback_results
            except Exception as e:
                logger.error(f"Error in fallback happy query: {str(e)}")
        
        # For lyrics search intent, try direct phrase match
        if intent == 'lyrics_search' and query_intent.get('song_title'):
            search_phrase = query_intent.get('song_title', '')
            
            logger.info(f"Trying phrase match for: '{search_phrase}'")
            phrase_results = await search_lyrics_by_phrase(search_phrase, top_k=1)
            
            # If we have vector results, combine them with phrase results
            if phrase_results and vector_results:
                combined_results = phrase_results + [r for r in vector_results if r not in phrase_results][:top_k-len(phrase_results)]
                logger.info(f"Combining {len(phrase_results)} phrase results with vector results")
                return combined_results
            elif phrase_results:
                logger.info(f"Found direct phrase match: '{phrase_results[0]['song_title']}'")
                return phrase_results
                
        # For song title search, try exact matching
        if query_intent.get('song_title'):
            title = query_intent.get('song_title')
            artist = query_intent.get('artist')
            
            logger.info(f"Trying exact song match: '{title}'{f' by {artist}' if artist else ''}")
            
            if artist:
                direct_results = await search_exact_song(title, artist)
            else:
                direct_results = await search_exact_song(title)
            
            # Combine with vector results if available
            if direct_results and vector_results:
                combined_results = []
                seen_ids = set()
                
                # First add direct results
                for result in direct_results:
                    if result['id'] not in seen_ids:
                        seen_ids.add(result['id'])
                        combined_results.append(result)
                
                # Then add vector results that aren't already included
                for result in vector_results:
                    if result['id'] not in seen_ids and len(combined_results) < top_k:
                        seen_ids.add(result['id'])
                        combined_results.append(result)
                
                logger.info(f"Combining {len(direct_results)} direct matches with vector results")
                return combined_results
            elif direct_results:
                return direct_results
        
        # For artist recommendations
        if intent == 'artist_recommendations' and query_intent.get('artist'):
            artist = query_intent.get('artist')
            logger.info(f"Finding songs for artist: '{artist}'")
            
            with engine.connect() as connection:
                artist_query = text("""
                SELECT song_id, track_name, track_artist, original_lyrics, cleaned_lyrics 
                FROM songs 
                WHERE LOWER(track_artist) LIKE LOWER(:artist_pattern)
                ORDER BY popularity_score DESC
                LIMIT :top_k
                """)
                
                results = connection.execute(artist_query, {
                    "artist_pattern": f"%{artist.lower()}%",
                    "top_k": top_k
                }).fetchall()
                
                for row in results:
                    intent_results.append({
                        "id": row.song_id,
                        "song_title": row.track_name,
                        "artist": row.track_artist,
                        "processed_text": row.original_lyrics,
                        "cleaned_text": row.cleaned_lyrics,
                        "similarity": 0.95,
                        "match_type": "artist"
                    })
                
                # Combine with vector results if available
                if intent_results and vector_results:
                    combined_results = []
                    seen_ids = set()
                    
                    # First add artist-specific results
                    for result in intent_results:
                        if result['id'] not in seen_ids:
                            seen_ids.add(result['id'])
                            combined_results.append(result)
                    
                    # Then add vector results that aren't already included
                    for result in vector_results:
                        if result['id'] not in seen_ids and len(combined_results) < top_k:
                            seen_ids.add(result['id'])
                            combined_results.append(result)
                    
                    logger.info(f"Combining {len(intent_results)} artist results with vector results")
                    return combined_results
                elif intent_results:
                    return intent_results
        
        # Return vector results if they exist (even if low confidence)
        if vector_results:
            logger.info(f"Using {len(vector_results)} vector results as primary results")
            return vector_results
            
        # Return any available results in priority order
        logger.info("No vector results, returning any available results")
        if phrase_results:
            return phrase_results
        if direct_results:
            return direct_results
        if audio_feature_results:
            return audio_feature_results
        if intent_results:
            return intent_results
        
        # Last resort - if it's a dance or happy query and nothing worked, just return random songs
        if is_dance_query or is_happy_query:
            logger.info("No results found but this is a mood query - returning random songs")
            try:
                with engine.connect() as connection:
                    random_query = text("""
                    SELECT song_id, track_name, track_artist, original_lyrics, cleaned_lyrics
                    FROM songs
                    ORDER BY RAND()
                    LIMIT :top_k
                    """)
                    
                    results = connection.execute(random_query, {"top_k": top_k}).fetchall()
                    
                    random_results = []
                    for row in results:
                        random_results.append({
                            "id": row.song_id,
                            "song_title": row.track_name,
                            "artist": row.track_artist,
                            "processed_text": row.original_lyrics,
                            "cleaned_text": row.cleaned_lyrics,
                            "similarity": 0.5,
                            "match_type": "random_fallback"
                        })
                    
                    if random_results:
                        return random_results
            except Exception as e:
                logger.error(f"Error in random fallback query: {str(e)}")
        
        return []
        
    except Exception as e:
        logger.error(f"Error in vector-first hybrid search: {str(e)}")
        return []
     

async def search_songs_with_ambiguity_detection(song_title, artist=None, limit=10):
    """
    Search for songs and detect if there are multiple artists for the same song title.
    Returns results grouped by unique combinations.
    """
    try:
        logger.info(f"Searching songs with ambiguity detection: '{song_title}'{' by ' + artist if artist else ''}")
        
        with engine.connect() as connection:
            if artist:
                # Search with artist specified
                query = text("""
                SELECT song_id, track_name, track_artist, original_lyrics, cleaned_lyrics,
                       popularity_score
                FROM songs 
                WHERE LOWER(track_name) LIKE LOWER(:song_pattern) 
                  AND LOWER(track_artist) LIKE LOWER(:artist_pattern)
                ORDER BY 
                    CASE WHEN LOWER(track_name) = LOWER(:exact_title) THEN 2 ELSE 1 END DESC,
                    CASE WHEN popularity_score IS NULL THEN 0 ELSE popularity_score END DESC
                LIMIT :limit
                """)
                
                results = connection.execute(query, {
                    "song_pattern": f"%{song_title}%",
                    "artist_pattern": f"%{artist}%",
                    "exact_title": song_title,
                    "limit": limit
                }).fetchall()
            else:
                # Search without artist - look for potential ambiguity
                query = text("""
                SELECT song_id, track_name, track_artist, original_lyrics, cleaned_lyrics,
                       popularity_score
                FROM songs 
                WHERE LOWER(track_name) LIKE LOWER(:song_pattern)
                ORDER BY 
                    CASE WHEN LOWER(track_name) = LOWER(:exact_title) THEN 2 ELSE 1 END DESC,
                    CASE WHEN popularity_score IS NULL THEN 0 ELSE popularity_score END DESC
                LIMIT :limit
                """)
                
                results = connection.execute(query, {
                    "song_pattern": f"%{song_title}%",
                    "exact_title": song_title,
                    "limit": limit
                }).fetchall()
            
            # Process results and detect ambiguity
            songs = []
            exact_matches = {}  # Track exact title matches by artist
            
            for row in results:
                song_data = {
                    "id": row.song_id,
                    "song_title": row.track_name,
                    "artist": row.track_artist,
                    "processed_text": row.original_lyrics,
                    "cleaned_text": row.cleaned_lyrics,
                    "popularity_score": row.popularity_score or 0,
                    "match_type": "fuzzy"
                }
                
                # Check if this is an exact title match
                if row.track_name.lower() == song_title.lower():
                    song_data["match_type"] = "exact"
                    exact_matches[row.track_artist] = song_data
                
                songs.append(song_data)
            
            # Analyze for ambiguity
            ambiguity_info = {
                "is_ambiguous": False,
                "exact_matches": len(exact_matches),
                "unique_artists": list(exact_matches.keys()),
                "needs_clarification": False
            }
            
            # If we have multiple exact matches with different artists, it's ambiguous
            if len(exact_matches) > 1 and not artist:
                ambiguity_info["is_ambiguous"] = True
                ambiguity_info["needs_clarification"] = True
                logger.info(f"Ambiguous song detected: '{song_title}' found with artists: {ambiguity_info['unique_artists']}")
            
            return {
                "songs": songs,
                "ambiguity": ambiguity_info
            }
            
    except Exception as e:
        logger.error(f"Error in ambiguity detection search: {str(e)}")
        return {
            "songs": [],
            "ambiguity": {"is_ambiguous": False, "exact_matches": 0, "unique_artists": [], "needs_clarification": False}
        }
    
async def resolve_artist_from_followup(question, song_title, context):
    """
    Analyze a follow-up message to extract the artist name when user is clarifying
    which version of a song they want.
    """
    is_thai = countthai(question) / len(question) > 0.15 if question else False
    
    # Direct pattern matching for artist selection
    artist_patterns = {
        'th': [
            r'(?:ของ|โดย|จาก)\s*(.+)',  # "ของ Taylor Swift", "โดย Ed Sheeran"
            r'(.+)(?:\s+version|\s+cover)',  # "Taylor Swift version"
            r'^(.+)$'  # Just the artist name
        ],
        'en': [
            r'(?:by|from)\s+(.+)',  # "by Taylor Swift", "from Ed Sheeran"  
            r'(.+)(?:\s+version|\s+cover)',  # "Taylor Swift version"
            r'^(.+)$'  # Just the artist name
        ]
    }
    
    patterns = artist_patterns['th'] if is_thai else artist_patterns['en']
    
    for pattern in patterns:
        match = re.search(pattern, question.strip(), re.IGNORECASE)
        if match:
            potential_artist = match.group(1).strip()
            # Clean up common words
            cleanup_words = ['version', 'cover', 'เวอร์ชั่น', 'ของ', 'โดย', 'จาก', 'by', 'from']
            for word in cleanup_words:
                potential_artist = re.sub(r'\b' + re.escape(word) + r'\b', '', potential_artist, flags=re.IGNORECASE).strip()
            
            if potential_artist and len(potential_artist) > 1:
                logger.info(f"Extracted artist from follow-up: '{potential_artist}'")
                return potential_artist
    
    return None

async def parse_query_with_llm(question, context=None, max_retries=3):
    """
    Parse user query to extract intent, song title, artist, etc.
    Enhanced with better handling of Thai language and context.
    Improved for mixed-language lyrics display requests.
    """
    is_thai = countthai(question) / len(question) > 0.15 if question else False
        
    # Thai lyrics display patterns
    if "ขอเนื้อเพลง" in question or "เนื้อเพลง" in question:
        # Extract song and artist
        remaining_text = question.replace("ขอเนื้อเพลง", "").replace("เนื้อเพลง", "").strip()
        
        # Pattern: ขอเนื้อเพลง [SONG] ของ [ARTIST]
        if "ของ" in remaining_text:
            parts = remaining_text.split("ของ")
            song_title = parts[0].strip()
            artist = parts[1].strip() if len(parts) > 1 else None
            
            logger.info(f"Direct pattern match: Thai lyrics display for song '{song_title}' by '{artist}'")
            return {
                'intent': 'lyrics_display',
                'song_title': song_title,
                'artist': artist,
                'original_query': question,
                'top_k': 1,
                'detected_language': 'th',
                'mood': None
            }
        else:
            # Just song title without artist
            song_title = remaining_text.strip()
            logger.info(f"Direct pattern match: Thai lyrics display for song '{song_title}'")
            return {
                'intent': 'lyrics_display',
                'song_title': song_title,
                'artist': None,
                'original_query': question,
                'top_k': 1,
                'detected_language': 'th',
                'mood': None
            }
    
    # English or mixed language lyrics display patterns
    english_lyrics_patterns = ["lyrics of", "lyrics to", "lyrics for", "lyrics", "words to"]
    for pattern in english_lyrics_patterns:
        if pattern in question.lower():
            # Extract song and artist
            remaining_text = question.lower().replace(pattern, "").strip()
            
            # Pattern: lyrics of [SONG] by [ARTIST]
            artist = None
            song_title = remaining_text
            
            # Check for artist indicators
            for artist_indicator in ["by", "from", "of"]:
                if f" {artist_indicator} " in remaining_text:
                    parts = remaining_text.split(f" {artist_indicator} ")
                    song_title = parts[0].strip()
                    artist = parts[1].strip() if len(parts) > 1 else None
                    break
            
            logger.info(f"Direct pattern match: English lyrics display for song '{song_title}' by '{artist}'")
            return {
                'intent': 'lyrics_display',
                'song_title': song_title,
                'artist': artist,
                'original_query': question,
                'top_k': 1,
                'detected_language': 'en' if not is_thai else 'th',
                'mood': None
            }
    
    # Extract context information
    context_str = ""
    previous_song = None
    previous_artist = None
    if context and len(context) > 0:
        context_str = "Previous conversation:\n"
        for i, ctx in enumerate(context[-3:]):
            context_str += f"- Query {i+1}: '{ctx.get('query')}'\n"
            if 'ambiguous_song' in ctx:
                context_str += f"  (This was asking for clarification about song: {ctx.get('ambiguous_song')})\n"
                previous_song = ctx.get('ambiguous_song')
            if 'artist' in ctx and ctx['artist']:
                previous_artist = ctx.get('artist')
                context_str += f"  (Mentioned artist: {previous_artist})\n"
            if 'song_title' in ctx and ctx['song_title']:
                previous_song = ctx.get('song_title')
                context_str += f"  (Mentioned song: {previous_song})\n"
    
    # Check if this might be a follow-up
    follow_up_indicator = False
    if previous_song:
        if is_thai and ("ของ" in question or "จาก" in question):
            tokens = word_tokenize(question, engine="newmm")
            request_tokens = [token for token in tokens if token.lower() in THAI_REQUEST_WORDS]
            if len(request_tokens) / len(tokens) > 0.3:
                follow_up_indicator = True
                logger.info(f"Detected Thai follow-up query for song: {previous_song}")
        elif not is_thai and any(x in question.lower() for x in ["by", "version", "artist", "sing", "sung"]):
            follow_up_indicator = True
            logger.info(f"Detected English follow-up query for song: {previous_song}")
    
    try:
        # Format system prompt with stronger constraints for intent understanding
        system_prompt = """You are a query understanding assistant specialized in music queries in Thai and English.
        CRITICAL INSTRUCTIONS FOR RESPONSE FORMAT:
        - Your response MUST be VALID JSON only
        - Do not include ANY text before or after the JSON
        
        CRITICAL INSTRUCTIONS FOR INTENT DETECTION:
        1. Pay attention to distinguish between two different types of lyrics requests:
           - For "what song has these lyrics?" → Use intent "lyrics_search" (identify a song from lyrics)
           - For "show me the lyrics to Song X" → Use intent "lyrics_display" (show the full lyrics)
        2. For lyrics identification queries, set intent to "lyrics_search" and extract the lyrics_fragment
        3. For lyrics display requests, set intent to "lyrics_display" and extract the song_title and artist
        4. For Thai queries like "เนื้อเพลงท่อน X คือเพลงอะไร", extract X as lyrics_fragment
        5. For Thai queries with phrases like "คือเพลงของอะไร", "เป็นเพลงอะไร", extract the lyrics before these phrases
        6. For Thai requests like "ขอเนื้อเพลง X" or "อยากรู้เนื้อเพลง X", use intent "lyrics_display"
        7. If request is for artist recommendations, set intent to "artist_recommendations"
        8. For mood-based searches like happy or dance songs, set intent to "mood_search"
        9. Default to "general_query" intent when unsure
        
        CRITICAL INSTRUCTIONS FOR SONG TITLE AND ARTIST EXTRACTION:
        1. For lyrics display requests in Thai like "ขอเนื้อเพลง X ของ Y":
           - Extract X as song_title and Y as artist
        2. For mixed-language requests like "ขอเนื้อเพลง [English Song] ของ [English Artist]":
           - Correctly identify the English song_title and artist
        3. ALWAYS prioritize extracting the song title and artist name exactly as they appear in the query
        4. Do not truncate or modify the song title or artist name unless removing request words
        
        CRITICAL INSTRUCTIONS FOR THAI LANGUAGE:
        1. Always identify Thai phrases like "เพลง", "เนื้อเพลง" as REQUEST INDICATORS, not song titles
        2. Remove words like "เนื้อเพลง", "ท่อน" from extracted lyrics fragments
        3. Be extremely careful to avoid including Thai request words in the song_title field
        4. Extract lyrics cleanly without request words for lyrics_search intents
        
        CRITICAL INSTRUCTIONS FOR MOOD DETECTION:
        1. For requests about "dance", "เต้น", "แดนซ์" set mood to "dance"
        2. For requests about "happy", "สนุก", "มีความสุข" set mood to "happy"
        3. For requests about "sad", "เศร้า" set mood to "sad"
        """
        
        if follow_up_indicator:
            user_prompt = f"""
            {context_str}
            Current query: "{question}"
            
            IMPORTANT: This appears to be a follow-up to a previous query about song "{previous_song}".
            The user is likely specifying which artist's version they want.
            
            Extract the artist name from this follow-up query and set the intent to "continuation".
            
            IMPORTANT: Respond with ONLY the JSON object. No explanations, no markdown, JUST the JSON.
            
            {{
                "song_title": "{previous_song}",
                "artist": "extracted artist or null",
                "intent": "continuation",
                "mood": null,
                "refers_to_previous": true
            }}
            """
        else:
            preprocessed = preprocess_thai_query(question)
            
            user_prompt = f"""
            {context_str}
            Current query: "{question}"
            PyThaiNLP preprocessed query: "{preprocessed}"

            SPECIAL FOCUS FOR THIS QUERY: If the query contains phrases like "ขอเนื้อเพลง" or mentions "lyrics" 
            along with a song name and possibly an artist, this is MOST LIKELY a lyrics_display request.
            
            Based on the query, determine the user's intent and extract relevant information.
            
            For lyrics display requests like "show me lyrics of song X" or Thai "ขอเนื้อเพลง X":
              - Set intent: "lyrics_display"
              - Extract song_title: the name of the song EXACTLY as it appears
              - Extract artist: the artist name EXACTLY if provided
            
            For requests like "lyrics of [Song] by [Artist]", "ขอเนื้อเพลง [Song] ของ [Artist]":
              - Set intent: "lyrics_display"
              - Extract song_title and artist exactly as they appear, preserving capitalization
            
            For lyrics identification requests like "what song has lyrics X" or Thai "X คือเพลงอะไร":
              - Set intent: "lyrics_search"
              - Extract lyrics_fragment: the actual lyrics portion
              - Do not include request words in lyrics_fragment
            
            For artist recommendations like "recommend songs by X":
              - Set intent: "artist_recommendations"
              - Extract artist name
            
            For mood-based requests like "recommend happy songs" or "เพลงเต้นดีๆ":
              - Set intent: "mood_search"
              - Set mood to the appropriate mood like "happy", "sad", "dance"
            
            For general questions about songs, set intent to "general_query".
            
            IMPORTANT: Respond with ONLY the JSON object. No explanations, no markdown, JUST the JSON.

            {{
                "song_title": null or "extracted title EXACTLY as it appears",
                "artist": null or "extracted artist EXACTLY as it appears",
                "intent": "lyrics_search|lyrics_display|song_info|artist_info|mood_search|artist_recommendations|continuation|general_query",
                "mood": null or "sad|happy|love|dance|etc",
                "lyrics_fragment": null or "extracted lyrics if this is a lyrics identification query",
                "refers_to_previous": true|false
            }}
            """
        
        response = ollama.chat(
            model='llama3.1',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.1}
        )
        
        # Extract and parse the JSON response
        content = response['message']['content']
        logger.debug(f"Raw LLM response: {content}")
        
        # Try to clean the response and find the JSON content
        cleaned_content = content.strip()
        
        # Try to find JSON between code block markers
        import re
        code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned_content, re.DOTALL)
        if code_block_match:
            cleaned_content = code_block_match.group(1).strip()
        
        # Find JSON content between curly braces
        json_start = cleaned_content.find('{')
        json_end = cleaned_content.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_content = cleaned_content[json_start:json_end]
            
            try:
                # Try to parse the JSON
                parsed_data = json.loads(json_content)
            except json.JSONDecodeError as je:
                logger.error(f"JSON decode error: {str(je)}")
                # Try to fix common JSON issues
                try:
                    fixed_json = json_content.replace("'", '"')
                    parsed_data = json.loads(fixed_json)
                    logger.info("Successfully fixed JSON by replacing single quotes")
                except json.JSONDecodeError:
                    try:
                        import ast
                        fixed_json = ast.literal_eval(json_content)
                        parsed_data = fixed_json
                        logger.info("Successfully fixed JSON with ast.literal_eval")
                    except:
                        raise ValueError(f"Invalid JSON in LLM response: {str(je)}")
            
            # Add language detection and original query
            parsed_data['detected_language'] = 'th' if is_thai else 'en'
            parsed_data['original_query'] = question
            parsed_data['top_k'] = 5
            parsed_data['preprocessed_query'] = preprocess_thai_query(question)
            
            # Convert any string "null" to actual null/None
            for key in parsed_data:
                if parsed_data[key] == "null":
                    parsed_data[key] = None
            
            # Process intent and related fields
            if parsed_data.get('intent') == 'lyrics_search' and parsed_data.get('lyrics_fragment'):
                logger.info(f"LLM identified lyrics search with fragment: '{parsed_data['lyrics_fragment']}'")
            elif parsed_data.get('intent') == 'lyrics_display' and parsed_data.get('song_title'):
                logger.info(f"LLM identified lyrics display request for song: '{parsed_data['song_title']}'")
            elif parsed_data.get('intent') == 'artist_recommendations' and parsed_data.get('artist'):
                logger.info(f"LLM identified artist recommendation request for: '{parsed_data['artist']}'")
            elif parsed_data.get('intent') == 'mood_search' and parsed_data.get('mood'):
                logger.info(f"LLM identified mood search for: '{parsed_data['mood']}'")
                
                # Double-check the mood against our keywords for dance/happy
                if 'dance' in parsed_data['mood'].lower() or any(word in question.lower() for word in ["dance", "dancing", "danceable", "เต้น", "แดนซ์"]):
                    parsed_data['mood'] = 'dance'
                elif 'happy' in parsed_data['mood'].lower() or any(word in question.lower() for word in ["happy", "cheerful", "สนุก", "มีความสุข"]):
                    parsed_data['mood'] = 'happy'
            
            return parsed_data
        else:
            logger.error("No valid JSON found in LLM response")
            raise ValueError("Invalid LLM response format")
            
    except Exception as e:
        logger.error(f"Error in LLM query parsing: {str(e)}")
        
        # Enhanced fallback for lyrics display intent specifically
        # Check for common lyrics display patterns with more specific matching
        if "lyrics" in question.lower():
            song_artist_match = re.search(r'lyrics\s+(?:of|to|for)?\s+(.+?)(?:\s+by\s+(.+))?$', question.lower())
            if song_artist_match:
                song_title = song_artist_match.group(1).strip()
                artist = song_artist_match.group(2).strip() if song_artist_match.group(2) else None
                
                return {
                    'intent': 'lyrics_display',
                    'song_title': song_title,
                    'artist': artist,
                    'original_query': question,
                    'top_k': 1,
                    'mood': None,
                    'detected_language': 'th' if is_thai else 'en'
                }
        
        if "ขอเนื้อเพลง" in question or "เนื้อเพลง" in question:
            # More robust Thai lyrics display pattern matching
            pattern_match = re.search(r'(?:ขอเนื้อเพลง|เนื้อเพลง)\s+(.+?)(?:\s+ของ\s+(.+))?$', question)
            if pattern_match:
                song_title = pattern_match.group(1).strip()
                artist = pattern_match.group(2).strip() if pattern_match.group(2) else None
                
                return {
                    'intent': 'lyrics_display',
                    'song_title': song_title,
                    'artist': artist,
                    'original_query': question,
                    'top_k': 1,
                    'mood': None,
                    'detected_language': 'th'
                }
        
        # Rest of the fallback logic from original function
        if any(word in question.lower() for word in ["dance", "dancing", "danceable", "เต้น", "แดนซ์", "ดานซ์"]):
            logger.info("Fallback detected dance request")
            return {
                'intent': 'mood_search',
                'song_title': None,
                'artist': None,
                'mood': 'dance',
                'original_query': question,
                'top_k': 5,
                'detected_language': 'th' if is_thai else 'en'
            }
        
        if any(word in question.lower() for word in ["happy", "happiness", "cheerful", "สนุก", "มีความสุข", "สุข"]):
            logger.info("Fallback detected happy request")
            return {
                'intent': 'mood_search',
                'song_title': None,
                'artist': None,
                'mood': 'happy',
                'original_query': question,
                'top_k': 5,
                'detected_language': 'th' if is_thai else 'en'
            }
        
        # Fall back to simple rule-based parsing for critical cases
        if is_thai and ("คือเพลงอะไร" in question or "ของเพลงอะไร" in question or "เป็นเพลงอะไร" in question):
            lyrics_part = question
            for phrase in ["คือเพลงอะไร", "ของเพลงอะไร", "เป็นเพลงอะไร", "เนื้อเพลง"]:
                if phrase in lyrics_part:
                    lyrics_part = lyrics_part.split(phrase)[0].strip()
            
            logger.info(f"Fallback identified lyrics search query with fragment: '{lyrics_part}'")
            return {
                'intent': 'lyrics_search',
                'song_title': None,
                'artist': None,
                'lyrics_fragment': lyrics_part,
                'original_query': question,
                'top_k': 5,
                'mood': None,
                'detected_language': 'th'
            }
    
    # If all else fails
    logger.warning("Using default rule-based query parsing")
    return {
        'intent': 'general_query',  
        'song_title': None,
        'artist': None,
        'original_query': question,
        'top_k': 5,
        'mood': None,
        'detected_language': 'th' if is_thai else 'en'
    }

async def generate_llm_response(prompt, original_system_prompt, max_retries=3):
    """
    Enhanced LLM response generator with improved language detection and greeting personalization.
    Focuses responses specifically on music-related queries and properly handles different languages
    and greeting styles.
    
    Parameters:
    -----------
    prompt : str
        The user prompt or question to be answered
    original_system_prompt : str
        The base system prompt that defines the bot's personality mode (buddy, mentor, fun)
    max_retries : int
        Number of attempts to make if there's an error with the LLM call
        
    Returns:
    --------
    str
        The generated response text
    """
    # Detect language using the proper function
    query_language = detect_language(prompt)
    is_thai = query_language == 'th'
    
    # Identify proper Thai greeting format - only for Thai language queries
    thai_greeting = "สวัสดีครับ"  # Default male polite form
    if is_thai:
        if "สวัสดีค่ะ" in prompt:
            thai_greeting = "สวัสดีค่ะ"  # Female polite form if user used it
        elif "สวัสดีครับ" in prompt:
            thai_greeting = "สวัสดีครับ"  # Male polite form if user used it
    
    # Enhance system prompt with music focus and proper greeting
    # Only use Thai greeting for Thai language, otherwise default to English greeting
    if is_thai:
        greeting_instruction = f"Always begin your response with '{thai_greeting}'"
        language_instruction = "Respond in Thai"
    else:
        greeting_instruction = "Always begin your response with 'Hello!'"
        language_instruction = "Respond in English"
    
    enhanced_system_prompt = f"""You are a music recommendation assistant specialized in songs, lyrics, and artists. 
    CRITICAL INSTRUCTIONS: 
    - {greeting_instruction}
    - ONLY provide information about music, songs, lyrics, and artists
    - If asked about topics unrelated to music, politely redirect to music-related questions
    - NEVER generate or make up song lyrics, only use what's provided in the context
    - If you don't have information about a requested song or artist, simply state it's not in your database
    - {language_instruction}
    - When recommending songs, respect the exact number the user requested (if specified)
    - Present lyrics as a single, uninterrupted block with proper separation
    
    CONTEXT AWARENESS INSTRUCTIONS:
    - Understand that short follow-up messages often refer to previous songs
    - Messages like "from [Artist]" or "จาก [Artist]" are asking about a previous song by that artist
    - If a query mentions an artist right after talking about a song, connect these as related
    - Treat "by [Artist]" or "โดย [Artist]" as referring to a previously mentioned song
    
    """ + original_system_prompt
    
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model='llama3.1', 
                messages=[
                    {'role': 'system', 'content': enhanced_system_prompt},
                    {'role': 'user', 'content': prompt},
                ],
                options={"temperature": 0.3}
            )
                
            if not response or 'message' not in response or 'content' not in response['message']:
                raise ValueError("Invalid LLM response structure")
            
            content = response['message']['content']
            
            # Verify the response starts with the correct greeting format
            # For Thai responses, make sure it starts with the proper Thai greeting
            if is_thai and not any(content.startswith(greeting) for greeting in [thai_greeting, "สวัสดี"]):
                content = f"{thai_greeting} {content}"
            # For English responses, make sure it starts with an English greeting
            elif not is_thai and not any(content.lower().startswith(greeting) for greeting in ["hello", "hi ", "hey ", "greetings"]):
                content = f"Hello! {content}"
                
            return content
            
        except Exception as e:
            logger.error(f"Error generating LLM response (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(sleep_time)
    
    # Fallback response in appropriate language
    if is_thai:
        return f"{thai_greeting} ขออภัย ฉันกำลังประมวลผลคำขอของคุณไม่สำเร็จ โปรดลองอีกครั้งในภายหลัง"
    return "Hello! I'm sorry, I'm having trouble processing your request right now. Please try again later."



async def search_with_vector(query_embedding, top_k=5):
    """
    Improved vector search that properly weights lyrics and metadata components
    to find the most relevant songs. This is the primary search method.
    """
    if not query_embedding:
        logger.error("Cannot perform search: No valid query embedding provided")
        return []
    
    vector_dim = len(query_embedding)
    
    try:
        with engine.connect() as connection:
            vector_json = json.dumps(query_embedding)
            
            # Enhanced query with better weighting between lyrics and metadata
            # Using a balanced approach for scoring
            query = text("""
            SELECT s.song_id, s.track_name, s.track_artist, 
                   s.original_lyrics, s.cleaned_lyrics,
                   1 - (v.lyrics_embedding <=> CAST(:vector_json AS VECTOR(:dim))) AS lyrics_similarity,
                   1 - (v.track_name_embedding <=> CAST(:vector_json AS VECTOR(:dim))) AS track_similarity,
                   1 - (v.artist_embedding <=> CAST(:vector_json AS VECTOR(:dim))) AS artist_similarity,
                   1 - (v.album_name_embedding <=> CAST(:vector_json AS VECTOR(:dim))) AS album_similarity,
                   (
                       0.6 * (1 - (v.lyrics_embedding <=> CAST(:vector_json AS VECTOR(:dim)))) + 
                       0.2 * (1 - (v.track_name_embedding <=> CAST(:vector_json AS VECTOR(:dim)))) +
                       0.15 * (1 - (v.artist_embedding <=> CAST(:vector_json AS VECTOR(:dim)))) +
                       0.05 * (1 - (v.album_name_embedding <=> CAST(:vector_json AS VECTOR(:dim))))
                   ) AS combined_score
            FROM songs s
            JOIN song_embeddings_vector v ON s.song_id = v.song_id
            ORDER BY combined_score DESC
            LIMIT :top_k
            """)
            
            results = connection.execute(query, {
                "vector_json": vector_json,
                "dim": vector_dim,
                "top_k": top_k
            }).fetchall()
            
            similarities = []
            for row in results:
                # Get individual similarity scores
                lyrics_sim = float(row.lyrics_similarity or 0)
                track_sim = float(row.track_similarity or 0)
                artist_sim = float(row.artist_similarity or 0)
                album_sim = float(row.album_similarity or 0)
                combined_score = float(row.combined_score or 0)
                
                # Find which similarity score is highest to determine match type
                similarities_dict = {
                    "vector_lyrics": lyrics_sim,
                    "vector_track": track_sim,
                    "vector_artist": artist_sim, 
                    "vector_album": album_sim
                }
                match_type = max(similarities_dict.items(), key=lambda x: x[1])[0]
                
                # Apply normalization to avoid artificially high scores
                if combined_score > 0.99:
                    combined_score = 0.85 + (combined_score - 0.99) * 0.15
                
                # Only include if there's some similarity
                if combined_score > 0.5:  # Lowered threshold to catch more potential matches
                    similarities.append({
                        "id": row.song_id,
                        "song_title": row.track_name,
                        "artist": row.track_artist,
                        "processed_text": row.original_lyrics,
                        "cleaned_text": row.cleaned_lyrics,
                        "similarity": combined_score,
                        "lyrics_similarity": lyrics_sim,
                        "track_similarity": track_sim,
                        "artist_similarity": artist_sim,
                        "album_similarity": album_sim,
                        "match_type": match_type
                    })
            
            return similarities
            
    except Exception as e:
        logger.error(f"Error in enhanced TiDB vector search: {str(e)}")
        return []
     
async def search_lyrics_by_phrase(phrase, top_k=1):
    try:
        with engine.connect() as connection:
            query = text("""
            SELECT song_id, track_name, track_artist, original_lyrics, cleaned_lyrics 
            FROM songs 
            WHERE LOWER(original_lyrics) LIKE LOWER(:phrase_pattern)
            OR LOWER(cleaned_lyrics) LIKE LOWER(:phrase_pattern)
            ORDER BY 
                CASE 
                    WHEN LOWER(original_lyrics) LIKE LOWER(:exact_pattern) THEN 3
                    WHEN LOWER(cleaned_lyrics) LIKE LOWER(:exact_pattern) THEN 3
                    ELSE 1
                END DESC,
                popularity_score DESC
            LIMIT :top_k
            """)
            
            results = connection.execute(query, {
                "phrase_pattern": f"%{phrase}%",
                "exact_pattern": f"%{phrase}%",
                "top_k": top_k
            }).fetchall()
            
            matches = []
            for row in results:
                score = 0.7
                if row.original_lyrics and phrase.lower() in row.original_lyrics.lower():
                    score = 0.98
                elif row.cleaned_lyrics and phrase.lower() in row.cleaned_lyrics.lower():
                    score = 0.95
                
                matches.append({
                    "id": row.song_id,
                    "song_title": row.track_name,
                    "artist": row.track_artist,
                    "processed_text": row.original_lyrics,
                    "cleaned_text": row.cleaned_lyrics,
                    "similarity": score,
                    "match_type": "exact_lyrics"
                })
            
            if matches:
                return matches
            
            # Try word-by-word search
            words = [w for w in phrase.split() if len(w) > 2]
            
            if words:
                word_conditions = []
                word_params = {}
                
                for i, word in enumerate(words):
                    word_conditions.append(f"LOWER(original_lyrics) LIKE :word_{i}")
                    word_params[f"word_{i}"] = f"%{word}%"
                
                word_query = text(f"""
                SELECT song_id, track_name, track_artist, original_lyrics, cleaned_lyrics,
                    (
                        {" + ".join([f"CASE WHEN LOWER(original_lyrics) LIKE :word_{i} THEN 1 ELSE 0 END" for i in range(len(words))])}
                    ) / {len(words)} AS match_score
                FROM songs
                WHERE {" OR ".join(word_conditions)}
                ORDER BY match_score DESC, popularity_score DESC
                LIMIT :top_k
                """)
                
                all_params = {**word_params, "top_k": top_k}
                word_results = connection.execute(word_query, all_params).fetchall()
                
                for row in word_results:
                    # Convert Decimal to float before arithmetic operations
                    match_score = float(getattr(row, 'match_score', 0))
                    if match_score > 0.4:
                        matches.append({
                            "id": row.song_id,
                            "song_title": row.track_name,
                            "artist": row.track_artist,
                            "processed_text": row.original_lyrics,
                            "cleaned_text": row.cleaned_lyrics,
                            "similarity": 0.7 + (match_score * 0.2),
                            "match_type": "partial_lyrics"
                        })
            
            return matches
    except Exception as e:
        logger.error(f"Error in lyrics phrase search: {str(e)}")
        return []
    
async def search_by_audio_features(mood, top_k=5):
    normalized_mood = THAI_MOOD_AUDIO_MAPPING.get(mood.lower(), mood.lower())
    mood_features = MOOD_AUDIO_FEATURES.get(normalized_mood)
    
    if not mood_features:
        return []
    
    try:
        with engine.connect() as connection:
            # Explicitly define fallback queries for common moods if no results
            # DANCE MOOD QUERY
            if normalized_mood == "dance":
                query = text("""
                SELECT s.song_id, s.track_name, s.track_artist, s.original_lyrics, s.cleaned_lyrics,
                       f.danceability, f.energy, f.valence
                FROM songs s
                JOIN song_features f ON s.song_id = f.song_id
                WHERE f.danceability > 0.0  -- Very permissive threshold for your data
                  AND f.energy > -0.3       -- Very permissive threshold for your data
                ORDER BY (f.danceability * 3.0 + f.energy * 2.0 + 
                         CASE WHEN s.popularity_score IS NULL THEN 0 ELSE s.popularity_score * 0.5 END) DESC
                LIMIT :top_k
                """)
            
            # HAPPY MOOD QUERY
            elif normalized_mood == "happy":
                query = text("""
                SELECT s.song_id, s.track_name, s.track_artist, s.original_lyrics, s.cleaned_lyrics,
                       f.danceability, f.energy, f.valence
                FROM songs s
                JOIN song_features f ON s.song_id = f.song_id
                WHERE f.valence > -0.3      -- Very permissive threshold for your data distribution
                  AND f.energy > -0.4       -- Very permissive threshold for your data distribution
                ORDER BY (f.valence * 3.0 + f.energy * 1.5 + f.danceability * 1.0 + 
                         CASE WHEN s.popularity_score IS NULL THEN 0 ELSE s.popularity_score * 0.5 END) DESC
                LIMIT :top_k
                """)
            
            # DEFAULT QUERY USING MOOD FEATURES
            else:
                # Build WHERE clauses based on mood features
                where_conditions = []
                for feature, criteria in mood_features.items():
                    if 'min' in criteria:
                        where_conditions.append(f"f.{feature} > {criteria['min']}")
                    if 'max' in criteria:
                        where_conditions.append(f"f.{feature} < {criteria['max']}")
                
                # Build ORDER BY expression (weighted sum of features)
                order_by_terms = []
                for feature, criteria in mood_features.items():
                    weight = criteria.get('weight', 1.0)
                    
                    if 'min' in criteria and 'max' not in criteria:
                        order_by_terms.append(f"f.{feature} * {weight}")
                    elif 'max' in criteria and 'min' not in criteria:
                        order_by_terms.append(f"(1 - f.{feature}) * {weight}")
                    elif 'min' in criteria and 'max' in criteria:
                        mid = (criteria['min'] + criteria['max']) / 2
                        order_by_terms.append(f"(1 - ABS(f.{feature} - {mid})) * {weight}")
                
                # Add popularity as a factor in ranking
                order_by_terms.append("CASE WHEN s.popularity_score IS NULL THEN 0 ELSE s.popularity_score * 0.5 END")
                
                where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
                order_by_clause = f"ORDER BY ({' + '.join(order_by_terms)}) DESC" if order_by_terms else "ORDER BY RAND()"
                
                # Build and execute the query
                query = text(f"""
                SELECT s.song_id, s.track_name, s.track_artist, s.original_lyrics, s.cleaned_lyrics,
                       f.danceability, f.energy, f.valence, f.acousticness, f.instrumentalness, f.tempo
                FROM songs s
                JOIN song_features f ON s.song_id = f.song_id
                {where_clause}
                {order_by_clause}
                LIMIT :top_k
                """)
            
            results = connection.execute(query, {"top_k": top_k}).fetchall()
            
            # If we got no results with the mood-specific query, try a more generic approach
            if not results:
                logger.info(f"No results using mood features for {normalized_mood}, trying generic query")
                
                # For dance/happy, try an extremely permissive query
                if normalized_mood in ["dance", "happy"]:
                    if normalized_mood == "dance":
                        generic_query = text("""
                        SELECT s.song_id, s.track_name, s.track_artist, s.original_lyrics, s.cleaned_lyrics,
                               f.danceability, f.energy, f.valence
                        FROM songs s
                        JOIN song_features f ON s.song_id = f.song_id
                        ORDER BY (f.danceability + f.energy) DESC, RAND()
                        LIMIT :top_k
                        """)
                    else:  # happy
                        generic_query = text("""
                        SELECT s.song_id, s.track_name, s.track_artist, s.original_lyrics, s.cleaned_lyrics,
                               f.danceability, f.energy, f.valence
                        FROM songs s
                        JOIN song_features f ON s.song_id = f.song_id
                        ORDER BY (f.valence + f.energy + f.danceability) DESC, RAND()
                        LIMIT :top_k
                        """)
                else:
                    # For other moods, just get random songs
                    generic_query = text("""
                    SELECT s.song_id, s.track_name, s.track_artist, s.original_lyrics, s.cleaned_lyrics,
                           f.danceability, f.energy, f.valence
                    FROM songs s
                    JOIN song_features f ON s.song_id = f.song_id
                    ORDER BY RAND()
                    LIMIT :top_k
                    """)
                
                results = connection.execute(generic_query, {"top_k": top_k}).fetchall()
                
                # If still no results, try direct from songs table without features
                if not results and (normalized_mood == "dance" or normalized_mood == "happy"):
                    logger.info(f"Still no results, trying direct from songs table for {normalized_mood}")
                    final_fallback_query = text("""
                    SELECT s.song_id, s.track_name, s.track_artist, s.original_lyrics, s.cleaned_lyrics
                    FROM songs s
                    ORDER BY RAND()
                    LIMIT :top_k
                    """)
                    results = connection.execute(final_fallback_query, {"top_k": top_k}).fetchall()
            
            songs = []
            for row in results:
                # Calculate a mood match score - higher value for happy/dance songs
                mood_score = 0.85  
                
                # Get audio features as a dictionary
                audio_features = {}
                for feature in ['danceability', 'energy', 'valence', 'acousticness', 'tempo']:
                    if hasattr(row, feature):
                        audio_features[feature] = getattr(row, feature)
                
                songs.append({
                    "id": row.song_id,
                    "song_title": row.track_name,
                    "artist": row.track_artist,
                    "processed_text": row.original_lyrics,
                    "cleaned_text": row.cleaned_lyrics,
                    "similarity": mood_score,
                    "match_type": "mood_feature",
                    "audio_features": audio_features
                })
            
            return songs
            
    except Exception as e:
        logger.error(f"Error in audio feature search: {str(e)}")
        return []
                             
async def search_exact_song(song_title, artist=None, limit=5):
    """
    Searches for exact song matches by title and optionally artist.
    Enhanced with better fuzzy matching and partial matching capabilities.
    """
    try:
        logger.info(f"Searching for song: '{song_title}'{' by ' + artist if artist else ''}")
        with engine.connect() as connection:
            # Step 1: Try exact match first (case insensitive)
            if artist:
                query = text("""
                SELECT song_id, track_name, track_artist, original_lyrics, cleaned_lyrics 
                FROM songs 
                WHERE LOWER(track_name) = LOWER(:exact_title) 
                  AND LOWER(track_artist) LIKE LOWER(:artist_pattern)
                ORDER BY CASE WHEN popularity_score IS NULL THEN 0 ELSE popularity_score END DESC
                LIMIT :limit
                """)
                
                results = connection.execute(query, {
                    "exact_title": song_title.lower(),
                    "artist_pattern": f"%{artist.lower()}%",
                    "limit": limit
                }).fetchall()
            else:
                query = text("""
                SELECT song_id, track_name, track_artist, original_lyrics, cleaned_lyrics 
                FROM songs 
                WHERE LOWER(track_name) = LOWER(:exact_title)
                ORDER BY CASE WHEN popularity_score IS NULL THEN 0 ELSE popularity_score END DESC
                LIMIT :limit
                """)
                
                results = connection.execute(query, {
                    "exact_title": song_title.lower(),
                    "limit": limit
                }).fetchall()
            
            if results:
                logger.info(f"Found exact match for '{song_title}'")
                return [{
                    "id": row.song_id,
                    "song_title": row.track_name,
                    "artist": row.track_artist,
                    "processed_text": row.original_lyrics,
                    "cleaned_text": row.cleaned_lyrics,
                    "similarity": 0.95,
                    "match_type": "direct"
                } for row in results]
            
            # Step 2: Try contains match if exact match fails
            logger.info(f"No exact match, trying contains match for '{song_title}'")
            if artist:
                contains_query = text("""
                SELECT song_id, track_name, track_artist, original_lyrics, cleaned_lyrics 
                FROM songs 
                WHERE LOWER(track_name) LIKE LOWER(:contains_title) 
                  AND LOWER(track_artist) LIKE LOWER(:artist_pattern)
                ORDER BY CASE WHEN popularity_score IS NULL THEN 0 ELSE popularity_score END DESC
                LIMIT :limit
                """)
                
                results = connection.execute(contains_query, {
                    "contains_title": f"%{song_title.lower()}%",
                    "artist_pattern": f"%{artist.lower()}%",
                    "limit": limit
                }).fetchall()
            else:
                contains_query = text("""
                SELECT song_id, track_name, track_artist, original_lyrics, cleaned_lyrics 
                FROM songs 
                WHERE LOWER(track_name) LIKE LOWER(:contains_title)
                ORDER BY CASE WHEN popularity_score IS NULL THEN 0 ELSE popularity_score END DESC
                LIMIT :limit
                """)
                
                results = connection.execute(contains_query, {
                    "contains_title": f"%{song_title.lower()}%",
                    "limit": limit
                }).fetchall()
            
            if results:
                logger.info(f"Found contains match for '{song_title}'")
                return [{
                    "id": row.song_id,
                    "song_title": row.track_name,
                    "artist": row.track_artist,
                    "processed_text": row.original_lyrics,
                    "cleaned_text": row.cleaned_lyrics,
                    "similarity": 0.9,
                    "match_type": "contains"
                } for row in results]
            
            # Step 3: Try word-by-word fuzzy matching if direct matching fails
            title_words = song_title.lower().split()
            # Only use meaningful words (longer than 2 characters)
            filter_words = [word for word in title_words if len(word) >= 3]
            
            if filter_words:
                logger.info(f"Trying fuzzy matching with words: {filter_words}")
                # Create OR conditions for each meaningful word
                word_conditions = []
                word_params = {}
                
                for i, word in enumerate(filter_words):
                    word_conditions.append(f"LOWER(track_name) LIKE :word_{i}")
                    word_params[f"word_{i}"] = f"%{word}%"
                
                fuzzy_query = text(f"""
                SELECT song_id, track_name, track_artist, original_lyrics, cleaned_lyrics,
                    (
                        {" + ".join([f"CASE WHEN LOWER(track_name) LIKE :word_{i} THEN 1 ELSE 0 END" for i in range(len(filter_words))])}
                    ) / {len(filter_words)} AS match_score
                FROM songs
                WHERE {" OR ".join(word_conditions)}
                """ + (f"AND LOWER(track_artist) LIKE :artist_pattern" if artist else "") + """
                ORDER BY match_score DESC, 
                    CASE WHEN popularity_score IS NULL THEN 0 ELSE popularity_score END DESC
                LIMIT :limit
                """)
                
                all_params = {**word_params, "limit": limit}
                if artist:
                    all_params["artist_pattern"] = f"%{artist.lower()}%"
                    
                fuzzy_results = connection.execute(fuzzy_query, all_params).fetchall()
                
                if fuzzy_results:
                    logger.info(f"Found {len(fuzzy_results)} fuzzy matches for '{song_title}'")
                    return [{
                        "id": row.song_id,
                        "song_title": row.track_name,
                        "artist": row.track_artist,
                        "processed_text": row.original_lyrics,
                        "cleaned_text": row.cleaned_lyrics,
                        "similarity": float(getattr(row, 'match_score', 0.7)),
                        "match_type": "fuzzy"
                    } for row in fuzzy_results]
            
            # If we get here, no matches were found
            logger.warning(f"No matches found for song: '{song_title}'{' by ' + artist if artist else ''}")
            return []
    
    except Exception as e:
        logger.error(f"Error in exact song search: {str(e)}")
        return []
    
async def format_lyrics_with_llm(raw_lyrics, song_title=None, artist=None):
    """
    Use the LLM to format lyrics properly, but WITHOUT changing the actual lyrics content.
    This function takes raw lyrics and asks the LLM only to add proper line breaks and structure.
    
    Args:
        raw_lyrics (str): The unformatted or poorly formatted lyrics text
        song_title (str, optional): Title of the song for context
        artist (str, optional): Artist name for context
        
    Returns:
        str: Properly formatted lyrics with clear verse separation
    """
    import re
    
    if not raw_lyrics or raw_lyrics.strip() == "":
        return ""
    
    # First, clean up any obvious duplication
    # If the lyrics are very long, they might be duplicated - let's detect this
    cleaned_lyrics = raw_lyrics
    if len(raw_lyrics) > 1000:
        # Try to detect where the song might repeat completely
        half_length = len(raw_lyrics) // 2
        first_chunk = raw_lyrics[:half_length]
        if first_chunk in raw_lyrics[half_length:]:
            cleaned_lyrics = first_chunk
    
    # Next, use the LLM to format the lyrics with proper line breaks and structure
    # but with strict instructions not to change ANY of the actual words
    prompt = f"""
    I need you to format these song lyrics to make them easier to read. 
    
    IMPORTANT INSTRUCTIONS:
    1. DO NOT change ANY of the words or content
    2. DO NOT add or remove ANY lyrics
    3. DO NOT try to correct or improve the lyrics
    4. ONLY add proper line breaks and verse/chorus separation
    5. Don't label sections as "Verse" or "Chorus" - just use spacing
    6. Use blank lines to separate different verses/sections
    7. Keep chorus lines together (like "run away, but we're running in circles")
    8. Join very short split lines that should be together (like "run away")
    9. Format for readability with a clear structure
    10. Don't explain what you're doing, just return the formatted lyrics
    
    {"Song: " + song_title if song_title else ""}
    {"Artist: " + artist if artist else ""}
    
    HERE ARE THE LYRICS TO FORMAT (DO NOT CHANGE ANY WORDS):
    ```
    {cleaned_lyrics}
    ```
    
    Return ONLY the formatted lyrics with no additional text:
    """
    
    # Use the LLM just for formatting, with a very strict system prompt
    system_prompt = """You are a lyrics formatter. Your ONLY job is to format lyrics with proper line breaks 
    and structure without changing any words. Do NOT add, remove, or change ANY words in the lyrics. 
    Do NOT add section labels. ONLY format the structure with line breaks for readability.
    Return ONLY the formatted lyrics with no explanations or additional text.
    """
    
    try:
        # Use the LLM to format the lyrics
        response_text = await generate_llm_response(prompt, system_prompt)
        
        # Extract just the formatted lyrics (remove any explanatory text the LLM might add)
        formatted_lyrics = response_text
        
        # Remove any markdown code blocks if present
        formatted_lyrics = re.sub(r'```.*?\n', '', formatted_lyrics)
        formatted_lyrics = re.sub(r'```', '', formatted_lyrics)
        
        # Remove any "Formatted lyrics:" or similar text
        prefixes = ["formatted lyrics:", "here are the formatted lyrics:", "lyrics:"]
        for prefix in prefixes:
            if formatted_lyrics.lower().startswith(prefix):
                formatted_lyrics = formatted_lyrics[len(prefix):].strip()
        
        # Remove any introduction that might have been added
        if song_title:
            formatted_lyrics = re.sub(rf".*{re.escape(song_title)}.*\n", "", formatted_lyrics, flags=re.IGNORECASE)
        if artist:
            formatted_lyrics = re.sub(rf".*{re.escape(artist)}.*\n", "", formatted_lyrics, flags=re.IGNORECASE)
        
        # Ensure there are no excessive blank lines (no more than 2 consecutive blank lines)
        formatted_lyrics = re.sub(r'\n{3,}', '\n\n', formatted_lyrics)
        
        # Add a fallback in case the LLM fails to format properly
        if not formatted_lyrics or len(formatted_lyrics) < len(cleaned_lyrics) / 2:
            # If the LLM returned significantly less content, it might have cut off the lyrics
            # Fall back to a basic formatting approach
            formatted_lyrics = basic_lyrics_formatter(cleaned_lyrics)
            
        return formatted_lyrics.strip()
    
    except Exception as e:
        logger.error(f"Error formatting lyrics with LLM: {str(e)}")
        # Fallback to basic formatting if LLM fails
        return basic_lyrics_formatter(cleaned_lyrics)


def basic_lyrics_formatter(lyrics):
    """
    Basic fallback formatter in case the LLM formatter fails.
    Uses simple rules to format lyrics.
    """
    import re
    
    # Add line breaks after punctuation and common line endings
    formatted = lyrics.replace('. ', '.\n').replace('! ', '!\n').replace('? ', '?\n')
    
    # Split into lines of reasonable length (max ~50 chars)
    words = formatted.split()
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        current_text = ' '.join(current_line)
        
        # If line is long enough or contains a newline, start a new line
        if len(current_text) > 50 or '\n' in word:
            # Remove the newline character from the word if it exists
            if '\n' in word:
                current_line[-1] = word.replace('\n', '')
            
            lines.append(' '.join(current_line))
            current_line = []
    
    # Add any remaining words
    if current_line:
        lines.append(' '.join(current_line))
    
    # Join lines with newlines
    result = '\n'.join(lines)
    
    # Clean up any excessive spaces or newlines
    result = re.sub(r' +', ' ', result)
    result = re.sub(r'\n+', '\n', result)
    
    # Add paragraph breaks at logical points (chorus markers for "Circles")
    for marker in ["run away, but we're running in circles", "seasons change and our love went cold"]:
        result = result.replace(marker, f"\n\n{marker}")
    
    return result


async def handle_lyrics_display(intent_data, mode, question):
    """
    Handle lyrics display requests using a template-based approach with LLM-based formatting.
    """
    start_time = time.time()
    song_title = intent_data.get('song_title')
    artist = intent_data.get('artist')
    query_language = intent_data.get('detected_language', 'en')
    

    # Search for the song with improved matching
    logger.info(f"Lyrics display request for song: '{song_title}'{' by ' + artist if artist else ''}")
    song_results = await search_exact_song(song_title, artist, limit=1)
    
    if not song_results:
        # Try searching without the artist if no results with artist
        if artist:
            logger.info(f"No results with artist, trying without artist for: '{song_title}'")
            song_results = await search_exact_song(song_title, None, limit=1)
    
    if song_results:
        # Song found, generate response with lyrics
        song_info = song_results[0]
        
        # Check if the found song title matches the requested one (case insensitive)
        if song_info.get('song_title', '').lower() != song_title.lower():
            logger.warning(f"Found song {song_info['song_title']} doesn't match requested song {song_title}")
            
            # Return a message that the exact song wasn't found but a similar one was
            if query_language == 'th':
                greeting = "สวัสดีครับ"
                if "สวัสดีค่ะ" in question:
                    greeting = "สวัสดีค่ะ"
                
                if mode == "fun":
                    response_text = f"{greeting} 🎵 ฉันไม่พบเพลงชื่อ \"{song_title}\" แต่พบเพลงที่ใกล้เคียงคือ \"{song_info['song_title']}\" โดย {song_info['artist']} 🔍 คุณต้องการเนื้อเพลงนี้หรือไม่? ถ้าต้องการ โปรดถามใหม่โดยระบุชื่อเพลงนี้!"
                else:
                    response_text = f"{greeting} ขออภัย ฉันไม่พบเพลงชื่อ \"{song_title}\" แต่พบเพลงที่ใกล้เคียงคือ \"{song_info['song_title']}\" โดย {song_info['artist']} หากคุณต้องการเนื้อเพลงนี้ โปรดถามใหม่โดยระบุชื่อเพลงนี้"
            else:
                if mode == "fun":
                    response_text = f"Hello there! 🎵 I couldn't find a song titled \"{song_title}\", but I found a similar one: \"{song_info['song_title']}\" by {song_info['artist']} 🔍 Would you like lyrics for this song instead? If so, please ask again with this exact title!"
                else:
                    response_text = f"Hello! I couldn't find a song titled \"{song_title}\", but I found a similar one: \"{song_info['song_title']}\" by {song_info['artist']}. If you'd like lyrics for this song, please ask again with this exact title."
            
            return {
                "response": response_text,
                "mode": mode,
                "intent": "lyrics_display",
                "sources": [
                    {
                        "title": song_info["song_title"],
                        "artist": song_info["artist"],
                        "similarity": 0.8,
                        "match_type": "similar_song"
                    }
                ],
                "processing_time": round(time.time() - start_time, 2)
            }
        
        # Format lyrics nicely
        raw_lyrics = song_info.get('processed_text', '').replace('\\n', '\n').replace('\\r', '\r')
        
        if not raw_lyrics or raw_lyrics.strip() == "":
            # Song found but no lyrics
            if query_language == 'th':
                greeting = "สวัสดีครับ"
                if "สวัสดีค่ะ" in question:
                    greeting = "สวัสดีค่ะ"
                    
                if mode == "fun":
                    response_text = f"{greeting} 🎵 ฉันพบเพลง \"{song_info['song_title']}\" โดย {song_info['artist']} แล้ว แต่น่าเสียดายที่ไม่มีเนื้อเพลงในฐานข้อมูลของเรา 😔 ลองถามเพลงอื่นดูนะ!"
                else:
                    response_text = f"{greeting} ฉันพบเพลง \"{song_info['song_title']}\" โดย {song_info['artist']} แล้ว แต่ไม่มีเนื้อเพลงในฐานข้อมูลของเรา โปรดลองถามเพลงอื่น"
            else:
                if mode == "fun":
                    response_text = f"Hello there! 🎵 I found the song \"{song_info['song_title']}\" by {song_info['artist']}, but unfortunately there are no lyrics available in our database 😔 Try asking for another song!"
                else:
                    response_text = f"Hello! I found the song \"{song_info['song_title']}\" by {song_info['artist']}, but there are no lyrics available in our database. Please try asking for another song."
        else:
            # Format the lyrics using the LLM formatter
            formatted_lyrics = await format_lyrics_with_llm(
                raw_lyrics, 
                song_title=song_info['song_title'], 
                artist=song_info['artist']
            )
            
            # Determine appropriate greeting
            if query_language == 'th':
                greeting = "สวัสดีครับ"
                if "สวัสดีค่ะ" in question:
                    greeting = "สวัสดีค่ะ"
            else:
                greeting = "Hello!"
            
            # Use hard-coded intro and outro to avoid translation issues
            if query_language == 'th':
                if mode == "fun":
                    intro = f"เพลง \"{song_info['song_title']}\" ของ {song_info['artist']} เป็นเพลงที่มีเอกลักษณ์เฉพาะตัว และเป็นหนึ่งในผลงานที่โดดเด่นของเขา"
                    outro = "หวังว่าคุณจะชอบเพลงนี้!"
                elif mode == "mentor":
                    intro = f"เพลง \"{song_info['song_title']}\" ของ {song_info['artist']} เป็นผลงานที่มีความโดดเด่นด้านการเล่าเรื่องผ่านเนื้อเพลง"
                    outro = "ขอบคุณที่สนใจในเพลงนี้"
                else:  # buddy mode
                    intro = f"นี่คือเพลง \"{song_info['song_title']}\" ของ {song_info['artist']} ที่คุณขอมา"
                    outro = "ขอบคุณที่ฟัง"
            else:
                if mode == "fun":
                    intro = f"Here's the unique track \"{song_info['song_title']}\" by {song_info['artist']}, one of their standout works!"
                    outro = "Hope you enjoy this song!"
                elif mode == "mentor":
                    intro = f"The song \"{song_info['song_title']}\" by {song_info['artist']} features compelling storytelling through its lyrics"
                    outro = "Thank you for your interest in this track"
                else:  # buddy mode
                    intro = f"Here's \"{song_info['song_title']}\" by {song_info['artist']} that you requested"
                    outro = "Thanks for listening"
            
            # Now assemble the response using a template with the formatted lyrics
            if query_language == 'th':
                if mode == "fun":
                    emoji_song = "🎵"
                    emoji_artist = "🎤"
                    emoji_end = "🎶"
                    response_text = f"{greeting} {intro} {emoji_song}\n\n{emoji_song} เพลง: {song_info['song_title']}\n{emoji_artist} ศิลปิน: {song_info['artist']}\n\n{formatted_lyrics}\n\n{outro} {emoji_end}"
                else:
                    response_text = f"{greeting} {intro}\n\nเพลง: {song_info['song_title']}\nศิลปิน: {song_info['artist']}\n\n{formatted_lyrics}\n\n{outro}"
            else:
                if mode == "fun":
                    emoji_song = "🎵"
                    emoji_artist = "🎤"
                    emoji_end = "🎶"
                    response_text = f"{greeting} {intro} {emoji_song}\n\n{emoji_song} Song: {song_info['song_title']}\n{emoji_artist} Artist: {song_info['artist']}\n\n{formatted_lyrics}\n\n{outro} {emoji_end}"
                else:
                    response_text = f"{greeting} {intro}\n\nSong: {song_info['song_title']}\nArtist: {song_info['artist']}\n\n{formatted_lyrics}\n\n{outro}"
        
        return {
            "response": response_text,
            "mode": mode,
            "intent": "lyrics_display",
            "sources": [
                {
                    "title": song_info["song_title"],
                    "artist": song_info["artist"],
                    "similarity": 1.0,
                    "match_type": "exact_lyrics_display"
                }
            ],
            "processing_time": round(time.time() - start_time, 2)
        }
    else:
        # Song not found in database
        if query_language == 'th':
            greeting = "สวัสดีครับ"
            if "สวัสดีค่ะ" in question:
                greeting = "สวัสดีค่ะ"
                
            if mode == "fun":
                response_text = f"{greeting} 🎵 ขอโทษด้วย ฉันไม่พบเพลง \"{song_title}\"{' โดย ' + artist if artist else ''} ในฐานข้อมูลของเรา 🔍 ลองถามเพลงอื่นดูนะ!"
            else:
                response_text = f"{greeting} ขออภัย ฉันไม่พบเพลง \"{song_title}\"{' โดย ' + artist if artist else ''} ในฐานข้อมูลของเรา กรุณาลองถามเพลงอื่น"
        else:
            if mode == "fun":
                response_text = f"Hello there! 🎵 Sorry, I couldn't find the song \"{song_title}\"{' by ' + artist if artist else ''} in our database 🔍 Try asking for another song!"
            else:
                response_text = f"Hello! Sorry, I couldn't find the song \"{song_title}\"{' by ' + artist if artist else ''} in our database. Please try asking for another song."
        
        return {
            "response": response_text,
            "mode": mode,
            "intent": "lyrics_display",
            "sources": [],
            "processing_time": round(time.time() - start_time, 2)
        }

async def update_chat_response(chat_id: int, response: str, intent: str, request_id: str):
    """Update the chat entry with the final response"""
    try:
        db = SessionLocal()
        chat_entry = db.query(ChatHistory).filter(ChatHistory.id == chat_id).first()
        if chat_entry:
            chat_entry.response = response
            chat_entry.intent = intent
            db.commit()
            logger.info(f"Updated chat {chat_id} with response")
        db.close()
        
        # Mark request as completed
        request_status[request_id] = "completed"
        
        # Clean up ongoing requests
        for session_id, req_id in list(ongoing_requests.items()):
            if req_id == request_id:
                del ongoing_requests[session_id]
                break
                
    except Exception as e:
        logger.error(f"Error updating chat response: {str(e)}")
        request_status[request_id] = "failed"


async def generate_song_intro_outro(song_info, mode, language, greeting):
    """
    Generate only the introduction and outro for a song using the LLM.
    This function limits the LLM's role to only creating the brief comments
    before and after the lyrics, not handling the lyrics themselves.
    
    Returns a dictionary with 'intro' and 'outro' keys.
    """
    personality = PERSONALITY_MODES.get(mode, PERSONALITY_MODES["buddy"])
    
    # Adjust emoji usage based on mode
    emoji_usage = "high" if mode == "fun" else "moderate" if mode == "buddy" else "minimal"
    
    if language == 'th':
        prompt = f"""
        # คำแนะนำในการสร้างคำแนะนำเพลงและคำลงท้าย (ห้ามรวมส่วนนี้ในคำตอบของคุณ)

        คุณต้องสร้างเฉพาะส่วนแนะนำสั้นๆ และคำลงท้ายสำหรับเพลง:
        
        1. คำแนะนำเพลง: เขียนประโยคสั้นๆ 1-2 ประโยคเกี่ยวกับเพลงนี้ในแบบที่น่าสนใจ
        2. คำลงท้าย: ประโยคสั้นๆ เกี่ยวกับความหวังว่าผู้ใช้จะชอบเพลงนี้
        
        การใช้อิโมจิ: {"มาก" if emoji_usage == "high" else "ปานกลาง" if emoji_usage == "moderate" else "น้อย"}
        รูปแบบการตอบ: {personality["response_format"]}
        
        ข้อมูลเพลง:
        ชื่อเพลง: {song_info['song_title']}
        ศิลปิน: {song_info['artist']}
        
        # ข้อสำคัญมาก:
        - ตอบเป็น JSON ในรูปแบบ {{"intro": "คำแนะนำเพลง", "outro": "คำลงท้าย"}}
        - ตอบให้กระชับ แต่ละส่วนไม่เกิน 1-2 ประโยค
        - ห้ามพูดถึงเนื้อเพลงโดยตรง เพราะเรากำลังจะแสดงเนื้อเพลงแยกต่างหาก
        - ห้ามแนะนำเกี่ยวกับเพลงที่คุณไม่มีข้อมูลหรือแต่งขึ้นมาเอง
        """
    else:
        prompt = f"""
        # INSTRUCTIONS FOR GENERATING SONG INTRO AND OUTRO (DO NOT INCLUDE THIS SECTION IN YOUR ANSWER)
        
        Create only the brief introduction and outro for this song:
        
        1. Intro: Write 1-2 interesting sentences about the song (before showing lyrics)
        2. Outro: A brief closing sentence hoping the user enjoys the song (after showing lyrics)
        
        Emoji usage: {"High" if emoji_usage == "high" else "Moderate" if emoji_usage == "moderate" else "Minimal"}
        Response style: {personality["response_format"]}
        
        Song information:
        Title: {song_info['song_title']}
        Artist: {song_info['artist']}
        
        # CRITICAL NOTES:
        - Respond as a JSON in the format {{"intro": "your intro here", "outro": "your outro here"}}
        - Keep both sections brief, 1-2 sentences each
        - Do not directly reference the lyrics as they will be shown separately
        - Do not make claims about the song that you don't have information about
        """
    
    # Enhanced system prompt specifically for generating intro/outro
    system_prompt = f"""You are a music assistant providing brief song introductions and outros.
    {personality["system_prompt"]}
    
    CRITICAL INSTRUCTION: 
    - Respond ONLY with a JSON object containing 'intro' and 'outro' fields
    - Keep both parts brief (1-2 sentences)
    - Do not reference lyrics directly
    - Do not explain the format or process
    - Make only factual claims about well-known aspects of the song
    """
    
    try:
        # Generate introduction and outro with LLM
        response_text = await generate_llm_response(prompt, system_prompt)
        
        # Try to extract JSON
        import re
        import json
        
        # Look for JSON pattern
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                json_content = json_match.group(0)
                result = json.loads(json_content)
                
                # Validate that it has the required keys
                if 'intro' in result and 'outro' in result:
                    return result
            except:
                pass
        
        # If JSON parsing fails, extract intro and outro using pattern matching
        if language == 'th':
            # Fallback for Thai
            if 'intro' in response_text.lower() and 'outro' in response_text.lower():
                intro = re.search(r'"intro"\s*:\s*"([^"]+)"', response_text)
                outro = re.search(r'"outro"\s*:\s*"([^"]+)"', response_text)
                
                if intro and outro:
                    return {
                        'intro': intro.group(1),
                        'outro': outro.group(1)
                    }
            
            # Ultimate fallback - very simple intro/outro
            if mode == "fun":
                return {
                    'intro': f"นี่คือเพลง \"{song_info['song_title']}\" ที่ร้องโดย {song_info['artist']} ที่คุณขอมา!",
                    'outro': "หวังว่าคุณจะชอบเพลงนี้นะ!"
                }
            else:
                return {
                    'intro': f"นี่คือเนื้อเพลง \"{song_info['song_title']}\" โดย {song_info['artist']}",
                    'outro': "ขอบคุณที่ฟัง"
                }
        else:
            # Fallback for English
            if mode == "fun":
                return {
                    'intro': f"Here's the popular song \"{song_info['song_title']}\" by {song_info['artist']} that you requested!",
                    'outro': "Hope you enjoy this track!"
                }
            else:
                return {
                    'intro': f"Here are the lyrics to \"{song_info['song_title']}\" by {song_info['artist']}",
                    'outro': "Thanks for listening"
                }
    
    except Exception as e:
        logger.error(f"Error generating intro/outro: {str(e)}")
        # Default fallback values
        if language == 'th':
            return {
                'intro': f"นี่คือเนื้อเพลง \"{song_info['song_title']}\" โดย {song_info['artist']}",
                'outro': "ขอบคุณที่ฟัง"
            }
        else:
            return {
                'intro': f"Here are the lyrics to \"{song_info['song_title']}\" by {song_info['artist']}",
                'outro': "Thanks for listening"
            }

async def save_chat_history(message: ChatMessageCreate):
    try:
        db = SessionLocal()
        new_chat = ChatHistory(
            user_id=message.user_id,
            session_id=message.session_id,
            query=message.query,
            response=message.response,
            mode=message.mode,
            intent=message.intent
        )
        db.add(new_chat)
        db.commit()
        db.refresh(new_chat)
        db.close()
        return new_chat.to_dict()
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save chat history: {str(e)}")

async def get_chat_history_by_user(user_id: str, limit: int = 50, offset: int = 0):
    """Retrieve chat history for a specific user"""
    try:
        db = SessionLocal()
        chats = db.query(ChatHistory).filter(
            ChatHistory.user_id == user_id
        ).order_by(
            ChatHistory.created_at.desc()
        ).offset(offset).limit(limit).all()
        
        result = [chat.to_dict() for chat in chats]
        db.close()
        return result
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {str(e)}")

async def get_chat_history_by_session(session_id: str, limit: int = 50, offset: int = 0):
    """Retrieve chat history for a specific session"""
    try:
        db = SessionLocal()
        chats = db.query(ChatHistory).filter(
            ChatHistory.session_id == session_id
        ).order_by(
            ChatHistory.created_at.desc()
        ).offset(offset).limit(limit).all()
        
        result = [chat.to_dict() for chat in chats]
        db.close()
        return result
    except Exception as e:
        logger.error(f"Error retrieving session chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session chat history: {str(e)}")

@app.post("/save-chat", response_model=ChatMessageResponse)
async def save_chat(message: ChatMessageCreate):
    """Save a chat message to the database"""
    return await save_chat_history(message)

@app.get("/chat-history/session/{session_id}")
async def get_session_chat_history(session_id: str, limit: int = 50, offset: int = 0):
    """Get chat history for a specific session"""
    return await get_chat_history_by_session(session_id, limit, offset)

@app.get("/chat-history/{user_id}")
async def get_user_chat_history(user_id: str, limit: int = 50, offset: int = 0):
    """Get chat history for a specific user"""
    return await get_chat_history_by_user(user_id, limit, offset)

@app.get("/chat-history/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Check if there's an ongoing request for this session"""
    if session_id in ongoing_requests:
        request_id = ongoing_requests[session_id]
        status = request_status.get(request_id, "unknown")
        return {
            "has_ongoing_request": True,
            "request_id": request_id,
            "status": status
        }
    return {
        "has_ongoing_request": False
    }

@app.post("/ask")
async def ask_question(request: Request):
    try:
        # Get input data
        start_time = time.time()
        input_data = await request.json()
        
        if not input_data or "question" not in input_data:
            raise HTTPException(status_code=400, detail="Invalid input format. Provide 'question' in request body.")

        question = input_data.get("question", "").strip()
        mode = input_data.get("mode", "buddy").lower()
        session_id = input_data.get("session_id", "default")
        user_id = input_data.get("user_id", "anonymous") 
        
        logger.info(f"=== NEW REQUEST === '{question}' in {mode} mode from user {user_id}")
        
        # Check if this is a non-music related query
        music_related = await is_music_related_query(question)
        if not music_related:
            # ใช้ LLM สร้างคำตอบที่เป็นธรรมชาติ
            query_language = detect_language(question)
            mode = input_data.get("mode", "buddy").lower()
            personality = PERSONALITY_MODES.get(mode, PERSONALITY_MODES["buddy"])
            
            # สร้าง system prompt สำหรับการปฏิเสธคำถามที่ไม่เกี่ยวกับเพลง
            if query_language == 'th':
                thai_greeting = "สวัสดีครับ"
                if "สวัสดีค่ะ" in question:
                    thai_greeting = "สวัสดีค่ะ"
                    
                system_prompt = f"""คุณเป็น AI ผู้ช่วยเกี่ยวกับเพลง เนื้อเพลง และศิลปิน
        คุณมีหน้าที่อธิบายให้ผู้ใช้เข้าใจว่าคุณตอบได้เฉพาะคำถามเกี่ยวกับเพลงเท่านั้น

        คำแนะนำ:
        - เริ่มต้นด้วย "{thai_greeting}"
        - อธิบายว่าคุณเป็น AI เฉพาะด้านเพลง
        - บอกว่าไม่สามารถตอบคำถามเกี่ยวกับหอขิ้งอื่นได้
        - แนะนำให้ผู้ใช้ถามเกี่ยวกับเพลง ศิลปิน หรือเนื้อเพลง
        - ใช้รูปแบบการตอบที่เป็น {personality["response_format"]}
        - {"ใส่ emoji ประกอบ" if mode == "fun" else "พูดด้วยท่าทีเป็นมิตร"}"""

                user_prompt = f"""ผู้ใช้ถาม: "{question}"

        คำถามนี้ไม่เกี่ยวข้องกับเพลง เนื้อเพลง หรือศิลปิน
        กรุณาตอบเพื่ออธิบายว่าคุณเป็น AI เกี่ยวกับเพลงและไม่สามารถตอบคำถามเกี่ยวกับเรื่องอื่นได้
        แนะนำให้ผู้ใช้ถามเกี่ยวกับเพลงแทน"""

            else:
                system_prompt = f"""You are an AI assistant specialized in music, lyrics, and artists.
        Your job is to explain to users that you can only answer music-related questions.

        Instructions:
        - Always start with "Hello!"
        - Explain that you are a music-specialized AI
        - Say you cannot answer questions about other topics
        - Suggest the user ask about songs, artists, or lyrics instead
        - Use a {personality["response_format"]} tone
        - {"Include emojis to make it fun" if mode == "fun" else "Be friendly and helpful"}"""

                user_prompt = f"""User asked: "{question}"

        This question is not related to music, lyrics, or artists.
        Please respond explaining that you are a music AI and cannot answer questions about other topics.
        Suggest the user ask about music instead."""

            # สร้างคำตอบด้วย LLM
            response_text = await generate_llm_response(user_prompt, system_prompt)
            
            # Save this interaction to history
            if user_id != "anonymous":
                chat_data = ChatMessageCreate(
                    user_id=user_id,
                    session_id=session_id,
                    query=question,
                    response=response_text,
                    mode=mode,
                    intent="off_topic"
                )
                await save_chat_history(chat_data)
            
            return {
                "response": response_text,
                "mode": mode,
                "intent": "off_topic", 
                "sources": [],
                "processing_time": round(time.time() - start_time, 2)
            }
        # 1. Parse query intent using LLM
        context = conversation_context.get_context(session_id)
        intent_data = await parse_query_with_llm(question, context)
        
        # Extract requested song count for recommendation intents
        if intent_data.get('intent') == 'artist_recommendations':
            requested_count = extract_requested_song_count(question)
            if requested_count:
                intent_data['requested_song_count'] = requested_count
        
        # Update conversation context
        conversation_context.update_context(session_id, {
            'query': question,
            'intent': intent_data.get('intent'),
            'song_title': intent_data.get('song_title'),
            'artist': intent_data.get('artist'),
            'mood': intent_data.get('mood'),
            'requested_song_count': intent_data.get('requested_song_count', 5)
        })
        
        if intent_data.get('intent') == 'lyrics_display':
            logger.info(f"Handling lyrics display request for song: '{intent_data.get('song_title')}'")
            
            # Call the new handler function
            result = await handle_lyrics_display(intent_data, mode, question)
            
            # Save chat history for lyrics display
            if user_id != "anonymous":
                chat_data = ChatMessageCreate(
                    user_id=user_id,
                    session_id=session_id,
                    query=question,
                    response=result["response"],
                    mode=mode,
                    intent="lyrics_display"
                )
                await save_chat_history(chat_data)
            
            return result

        # Handle lyrics identification queries
        if intent_data.get('intent') == 'lyrics_search':
            # Get lyrics fragment from either field
            lyrics_fragment = intent_data.get('lyrics_fragment')
            if not lyrics_fragment and intent_data.get('song_title'):
                lyrics_fragment = intent_data.get('song_title')
                
            if lyrics_fragment:
                logger.info(f"Searching for lyrics fragment: '{lyrics_fragment}'")
                phrase_results = await search_lyrics_by_phrase(lyrics_fragment, top_k=3)
            
                if phrase_results:
                    # Format response with the identified song
                    song_info = phrase_results[0]  # Take the best match
                    query_language = intent_data.get('detected_language', 'en')
                    personality = PERSONALITY_MODES.get(mode, PERSONALITY_MODES["buddy"])
                    
                    if query_language == 'th':
                        greeting = "สวัสดีครับ"
                        if "สวัสดีค่ะ" in question:
                            greeting = "สวัสดีค่ะ"
                            
                        if mode == "fun":
                            response_text = f"{greeting} 🎵 เนื้อเพลงที่คุณถามมาเป็นของเพลง \"{song_info['song_title']}\" โดย {song_info['artist']} นั่นเองค่ะ! 🎤"
                        else:
                            response_text = f"{greeting} เนื้อเพลงที่คุณถามมาเป็นของเพลง \"{song_info['song_title']}\" โดย {song_info['artist']} ครับ"
                    else:
                        if mode == "fun":
                            response_text = f"Hello there! 🎵 The lyrics you asked about are from the song \"{song_info['song_title']}\" by {song_info['artist']}! 🎤"
                        else:
                            response_text = f"Hello! The lyrics you asked about are from the song \"{song_info['song_title']}\" by {song_info['artist']}."
                    
                    # Save chat history
                    if user_id != "anonymous":
                        chat_data = ChatMessageCreate(
                            user_id=user_id,
                            session_id=session_id,
                            query=question,
                            response=response_text,
                            mode=mode,
                            intent="lyrics_search"
                        )
                        await save_chat_history(chat_data)
                    
                    return {
                        "response": response_text,
                        "mode": mode,
                        "intent": "lyrics_search",
                        "sources": [
                            {
                                "title": item["song_title"], 
                                "artist": item["artist"], 
                                "similarity": round(item["similarity"], 3),
                                "match_type": item.get("match_type", "lyrics_match")
                            } 
                            for item in phrase_results
                        ],
                        "processing_time": round(time.time() - start_time, 2)
                    }
                else:
                    # No matching lyrics found
                    query_language = intent_data.get('detected_language', 'en')
                    
                    if query_language == 'th':
                        greeting = "สวัสดีครับ"
                        if "สวัสดีค่ะ" in question:
                            greeting = "สวัสดีค่ะ"
                            
                        if mode == "fun":
                            response_text = f"{greeting} 🎵 ขอโทษด้วยค่ะ ฉันไม่พบเพลงที่มีเนื้อร้องนี้ในฐานข้อมูลของเรา 🔍 ลองถามเนื้อเพลงอื่นดูนะคะ!"
                        else:
                            response_text = f"{greeting} ขออภัย ผมไม่พบเพลงที่มีเนื้อร้องตามที่คุณถามในฐานข้อมูลของเรา โปรดลองถามเนื้อเพลงอื่น"
                    else:
                        if mode == "fun":
                            response_text = "Hello there! 🎵 Sorry, I couldn't find any song with those lyrics in our database 🔍 Try asking about different lyrics!"
                        else:
                            response_text = "Hello! Sorry, I couldn't find any song with those lyrics in our database. Please try asking about different lyrics."
                    
                    # Save chat history
                    if user_id != "anonymous":
                        chat_data = ChatMessageCreate(
                            user_id=user_id,
                            session_id=session_id,
                            query=question,
                            response=response_text,
                            mode=mode,
                            intent="lyrics_search"
                        )
                        await save_chat_history(chat_data)
                    
                    return {
                        "response": response_text,
                        "mode": mode,
                        "intent": "lyrics_search",
                        "sources": [],
                        "processing_time": round(time.time() - start_time, 2)
                    }
        # 2. Handle artist recommendations intent 
        if intent_data.get('intent') == 'artist_recommendations' and intent_data.get('artist'):
            artist = intent_data.get('artist')
            query_language = intent_data.get('detected_language', 'en')
            limit = intent_data.get('requested_song_count', 5)
            
            result = await handle_artist_recommendations(artist, mode, query_language, limit=limit)
            result["processing_time"] = round(time.time() - start_time, 2)
            
            # Save chat history for artist recommendations
            if user_id != "anonymous":
                chat_data = ChatMessageCreate(
                    user_id=user_id,
                    session_id=session_id,
                    query=question,
                    response=result["response"],
                    mode=mode,
                    intent=intent_data['intent']
                )
                await save_chat_history(chat_data)
            
            return result
        
        # 3. Perform search based on intent - ALWAYS use vector search first, then fallback
        query_embedding = await get_query_embedding(question)
        similar_results = await hybrid_search(question, intent_data, query_embedding)

        # 4. Handle song analysis if requested
        if intent_data.get('intent') == 'song_analysis' and similar_results:
            song_data = similar_results[0]
            query_language = intent_data.get('detected_language', 'en')
            personality = PERSONALITY_MODES.get(mode, PERSONALITY_MODES["buddy"])  
            
            response_text = await generate_song_analysis(
                song_data,
                question,
                personality,
                query_language
            )
            
            # Save chat history for song analysis
            if user_id != "anonymous":
                chat_data = ChatMessageCreate(
                    user_id=user_id,
                    session_id=session_id,
                    query=question,
                    response=response_text,
                    mode=mode,
                    intent="song_analysis"
                )
                await save_chat_history(chat_data)
            
            return {
                "response": response_text,
                "mode": mode,
                "intent": "song_analysis",
                "sources": [
                    {
                        "title": item["song_title"], 
                        "artist": item["artist"], 
                        "similarity": round(item["similarity"], 3),
                        "match_type": item.get("match_type", "vector")
                    } 
                    for item in similar_results
                ],
                "processing_time": round(time.time() - start_time, 2)
            }

        # 5. Create context from results for LLM with more explicit information
        context_items = []
        sorted_results = sorted(similar_results, key=lambda x: x.get('similarity', 0), reverse=True)
        
        for item in sorted_results:
            processed_text = item.get('processed_text', '').replace('\\n', '\n').replace('\\r', '\r')
            
            if item == sorted_results[0]:
                context_items.append(
                    f"BEST MATCH: Song: {item['song_title']} by {item['artist']}\n"
                    f"(Match type: {item.get('match_type', 'unknown')}, Similarity score: {item.get('similarity', 0):.2f})\n"
                    f"Lyrics: {processed_text if processed_text.strip() else '[NO LYRICS FOUND IN DATABASE]'}"
                )
            else:
                context_items.append(
                    f"Song: {item['song_title']} by {item['artist']}\n"
                    f"(Match type: {item.get('match_type', 'unknown')}, Similarity score: {item.get('similarity', 0):.2f})\n"
                    f"Lyrics: {processed_text if processed_text.strip() else '[NO LYRICS FOUND IN DATABASE]'}"
                )
        
        context = "\n\n---\n\n".join(context_items)
        
        # 6. Generate response with LLM
        query_language = intent_data['detected_language']
        personality = PERSONALITY_MODES.get(mode, PERSONALITY_MODES["buddy"])
        
        # Check for greeting style in Thai
        thai_greeting = "สวัสดีครับ"
        if query_language == 'th' and "สวัสดีค่ะ" in question:
            thai_greeting = "สวัสดีค่ะ"
        
        if similar_results:
            # Check for ambiguous songs with same title by different artists
            if 'song_title' in intent_data and intent_data['song_title'] and not intent_data.get('artist'):
                if len(similar_results) > 1:
                    artists_list = [f"{item['artist']}" for item in similar_results]
                    artists_unique = list(set(artists_list))
                    
                    if len(artists_unique) > 1:
                        if query_language == 'th':
                            response_text = f"{thai_greeting} พบเพลง '{intent_data['song_title']}' จากหลายศิลปิน คุณต้องการเพลงของศิลปินคนไหน? ({', '.join(artists_unique[:5])})"
                        else:
                            response_text = f"Hello! I found the song '{intent_data['song_title']}' by multiple artists. Which artist's version would you like? ({', '.join(artists_unique[:5])})"
                        
                        conversation_context.update_context(session_id, {
                            'ambiguous_song': intent_data['song_title']
                        })
                        
                        # Save chat history for clarification
                        if user_id != "anonymous":
                            chat_data = ChatMessageCreate(
                                user_id=user_id,
                                session_id=session_id,
                                query=question,
                                response=response_text,
                                mode=mode,
                                intent="clarification"
                            )
                            await save_chat_history(chat_data)
                        
                        return {
                            "response": response_text,
                            "mode": mode,
                            "intent": "clarification",
                            "sources": [{"title": item["song_title"], "artist": item["artist"]} for item in similar_results],
                            "processing_time": round(time.time() - start_time, 2)
                        }
            
            # Normal response generation with enhanced prompts to never generate content
            if query_language == 'th':
                prompt = f"""
                คำถามของผู้ใช้: "{question}"
                
                ข้อมูลเพลงที่พบในฐานข้อมูล: 
                {context}
                
                คำแนะนำสำคัญ:
                - กรุณาตอบเป็นภาษาไทยในรูปแบบ {personality["response_format"]} ตามบุคลิกของคุณ
                - เริ่มต้นด้วยคำทักทาย: "{thai_greeting}"
                - แสดงเฉพาะข้อมูลและเนื้อเพลงที่ระบุไว้ข้างต้นเท่านั้น
                - ห้ามแต่งเนื้อเพลงหรือข้อมูลเพิ่มเติมที่ไม่มีอยู่ในข้อความข้างต้น
                - ถ้าไม่มีข้อมูลที่ผู้ใช้ถาม ให้แจ้งว่าไม่มีข้อมูลในฐานข้อมูล
                """
            else:
                prompt = f"""
                User query: "{question}"
                
                Songs information found in database:
                {context}
                
                Important instructions:
                - Please respond in English in a {personality["response_format"]} style.
                - Always start with a greeting: "Hello!"
                - ONLY display information and lyrics shown above.
                - DO NOT generate or make up any information not provided in the context.
                - If information isn't available in the context, simply state it's not in the database.
                """
            
            response_text = await generate_llm_response(prompt, personality["system_prompt"])
        else:
            # No results found
            if query_language == 'th':
                response_text = f"{thai_greeting} ขออภัย ไม่พบข้อมูลเพลงที่ตรงกับคำค้นหาของคุณในฐานข้อมูลของเรา โปรดลองค้นหาด้วยชื่อเพลงหรือชื่อศิลปินอื่น"
            else:
                response_text = "Hello! Sorry, I couldn't find any songs matching your query in our database. Please try searching with a different song title or artist name."
        
        # Save chat history for regular responses
        if user_id != "anonymous":
            chat_data = ChatMessageCreate(
                user_id=user_id,
                session_id=session_id,
                query=question,
                response=response_text,
                mode=mode,
                intent=intent_data['intent']
            )
            await save_chat_history(chat_data)
    
        # 7. Return the response
        return {
            "response": response_text,
            "mode": mode,
            "intent": intent_data['intent'],
            "sources": [
                {
                    "title": item["song_title"], 
                    "artist": item["artist"], 
                    "similarity": round(item["similarity"], 3),
                    "match_type": item.get("match_type", "vector")
                } 
                for item in similar_results
            ],
            "processing_time": round(time.time() - start_time, 2)
        }
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
# Run the app
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Optimized Music RAG API with Vector Search")
    uvicorn.run(app, host="0.0.0.0", port=8000)