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
    is_thai = countthai(question) / len(question) > 0.15 if question else False
    if is_thai:
        tokens = word_tokenize(question, engine="newmm")
        filtered_tokens = [token for token in tokens if token.lower() not in THAI_STOPWORDS]
        filtered_query = " ".join(filtered_tokens)
        logger.info(f"PyThaiNLP processed: '{question}' → '{filtered_query}'")
        return filtered_query
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
    using the LLM to generate personality-specific responses
    """
    start_time = time.time()
    
    # Get the personality configuration
    personality = PERSONALITY_MODES.get(mode, PERSONALITY_MODES["buddy"])
    
    # Log the request
    logger.info(f"Finding songs by artist: {artist} in {mode} mode")
    
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
        ผู้ใช้ขอคำแนะนำเพลงของ {artist}
        
        ข้อมูลเพลงที่พบในฐานข้อมูล:
        {song_list}
        
        กรุณาสร้างคำตอบที่มีรายละเอียดเกี่ยวกับเพลงของ {artist} พร้อมแนะนำเพลงเหล่านี้ ใช้รูปแบบการตอบที่เป็น {personality['response_format']} 
        อย่าพูดถึงเพลงอื่นนอกเหนือจากที่แสดงในรายการ และไม่ต้องมีข้อความว่ามีเพลงอื่นๆอีก
        """
    else:
        user_prompt = f"""
        The user asked for song recommendations by {artist}
        
        Songs found in database:
        {song_list}
        
        Please create a detailed response recommending these songs by {artist} in a {personality['response_format']} style.
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
        for song in all_songs
    ]
    
    # Final response object
    result = {
        "response": response_text,
        "processing_time": time.time() - start_time,
        "songs": all_songs,
        "intent": "artist_recommendations", 
        "sources": sources,
        "mode": mode
    }
    
    return result

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
     
async def parse_query_with_llm(question, context=None, max_retries=3):
    """
    Parse user query to extract intent, song title, artist, etc.
    Enhanced with better handling of Thai language and context.
    """
    is_thai = countthai(question) / len(question) > 0.15 if question else False
    
    # Direct detection for dance and happy queries (before calling LLM)
    # This improves quick detection of mood queries
    is_dance_query = False
    for word in ["dance", "dancing", "danceable", "เต้น", "แดนซ์", "ดานซ์", "เต้นรำ"]:
        if word.lower() in question.lower():
            is_dance_query = True
            logger.info(f"Directly detected dance request with keyword: '{word}'")
            return {
                'intent': 'mood_search',
                'song_title': None,
                'artist': None,
                'mood': 'dance',
                'original_query': question,
                'top_k': 5,
                'detected_language': 'th' if is_thai else 'en'
            }
    
    is_happy_query = False
    for word in ["happy", "happiness", "cheerful", "สนุก", "มีความสุข", "สุข", "ร่าเริง", "สนุกสนาน"]:
        if word.lower() in question.lower():
            is_happy_query = True
            logger.info(f"Directly detected happy request with keyword: '{word}'")
            return {
                'intent': 'mood_search',
                'song_title': None,
                'artist': None,
                'mood': 'happy',
                'original_query': question,
                'top_k': 5,
                'detected_language': 'th' if is_thai else 'en'
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
        1. Pay special attention to detect when users ask "what song has these lyrics" or similar questions
        2. For lyrics identification queries, set intent to "lyrics_search" and extract the lyrics_fragment
        3. For Thai queries like "เนื้อเพลงท่อน X คือเพลงอะไร" extract X as lyrics_fragment
        4. For Thai queries with phrases like "คือเพลงของอะไร", "เป็นเพลงอะไร", extract the lyrics before these phrases
        5. If request is for artist recommendations, set intent to "artist_recommendations"
        6. For mood-based searches like happy or dance songs, set intent to "mood_search"
        7. Default to "general_query" intent when unsure
        
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

            Based on the query, determine the user's intent and extract relevant information.
            
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
                "song_title": null or "extracted title",
                "artist": null or "extracted artist",
                "intent": "lyrics_search|song_info|artist_info|mood_search|artist_recommendations|continuation|general_query",
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
        
        # Enhanced fallback for mood searches
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
            # Very basic extraction for fallback
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
    enhanced_system_prompt = """You are a music recommendation assistant. 
    CRITICAL INSTRUCTION: 
    - ALWAYS begin your response with a greeting to the user
    - ONLY display information provided to you in the context
    - NEVER generate or make up song lyrics, even if you know the song
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
                
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Error generating LLM response (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(sleep_time)
    
    return "Sorry, I'm having trouble processing your request right now. Please try again later."

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
            LIMIT 1
            """)
            
            results = connection.execute(query, {
                "phrase_pattern": f"%{phrase}%",
                "exact_pattern": f"%{phrase}%"
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
                LIMIT 1
                """)
                
                word_results = connection.execute(word_query, word_params).fetchall()
                
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
                            "similarity": 0.7 + (match_score * 0.2),  # Now this will work
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
    Includes fuzzy matching as a fallback if exact matches aren't found.
    """
    try:
        with engine.connect() as connection:
            if artist:
                # Title + artist search with exact match
                query = text("""
                SELECT song_id, track_name, track_artist, original_lyrics, cleaned_lyrics 
                FROM songs 
                WHERE LOWER(track_name) = LOWER(:exact_title) 
                  AND LOWER(track_artist) LIKE LOWER(:artist_pattern)
                LIMIT :limit
                """)
                
                results = connection.execute(query, {
                    "exact_title": song_title.lower(),
                    "artist_pattern": f"%{artist.lower()}%",
                    "limit": limit
                }).fetchall()
            else:
                # Title-only search with exact match
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
                return [{
                    "id": row.song_id,
                    "song_title": row.track_name,
                    "artist": row.track_artist,
                    "processed_text": row.original_lyrics,
                    "cleaned_text": row.cleaned_lyrics,
                    "similarity": 0.95,
                    "match_type": "direct"
                } for row in results]
            
            # Try fuzzy matching with word similarity
            title_words = song_title.lower().split()
            filter_words = [word for word in title_words if len(word) >= 3]
            
            if filter_words:
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
                    return [{
                        "id": row.song_id,
                        "song_title": row.track_name,
                        "artist": row.track_artist,
                        "processed_text": row.original_lyrics,
                        "cleaned_text": row.cleaned_lyrics,
                        "similarity": float(getattr(row, 'match_score', 0.7)),
                        "match_type": "fuzzy"
                    } for row in fuzzy_results]
            
            return []
    
    except Exception as e:
        logger.error(f"Error in exact song search: {str(e)}")
        return []

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
        
        # 1. Parse query intent using LLM
        context = conversation_context.get_context(session_id)
        intent_data = await parse_query_with_llm(question, context)
        
        # Update conversation context
        conversation_context.update_context(session_id, {
            'query': question,
            'intent': intent_data.get('intent'),
            'song_title': intent_data.get('song_title'),
            'artist': intent_data.get('artist'),
            'mood': intent_data.get('mood')
        })
        
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
                    if mode == "fun":
                        response_text = f"สวัสดีค่ะ! 🎵 เนื้อเพลงที่คุณถามมาเป็นของเพลง \"{song_info['song_title']}\" โดย {song_info['artist']} นั่นเองค่ะ! 🎤"
                    else:
                        response_text = f"สวัสดีครับ เนื้อเพลงที่คุณถามมาเป็นของเพลง \"{song_info['song_title']}\" โดย {song_info['artist']} ครับ"
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
                    if mode == "fun":
                        response_text = "สวัสดีค่ะ! 🎵 ขอโทษด้วยค่ะ ฉันไม่พบเพลงที่มีเนื้อร้องนี้ในฐานข้อมูลของเรา 🔍 ลองถามเนื้อเพลงอื่นดูนะคะ!"
                    else:
                        response_text = "สวัสดีครับ ขออภัย ผมไม่พบเพลงที่มีเนื้อร้องตามที่คุณถามในฐานข้อมูลของเรา โปรดลองถามเนื้อเพลงอื่น"
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
            
            result = await handle_artist_recommendations(artist, mode, query_language, limit=5)
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
        
        if similar_results:
            # Check for ambiguous songs with same title by different artists
            if 'song_title' in intent_data and intent_data['song_title'] and not intent_data.get('artist'):
                if len(similar_results) > 1:
                    artists_list = [f"{item['artist']}" for item in similar_results]
                    artists_unique = list(set(artists_list))
                    
                    if len(artists_unique) > 1:
                        if query_language == 'th':
                            response_text = f"พบเพลง '{intent_data['song_title']}' จากหลายศิลปิน คุณต้องการเพลงของศิลปินคนไหน? ({', '.join(artists_unique[:5])})"
                        else:
                            response_text = f"I found the song '{intent_data['song_title']}' by multiple artists. Which artist's version would you like? ({', '.join(artists_unique[:5])})"
                        
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
                - ONLY display information and lyrics shown above.
                - DO NOT generate or make up any information not provided in the context.
                - If information isn't available in the context, simply state it's not in the database.
                """
            
            response_text = await generate_llm_response(prompt, personality["system_prompt"])
        else:
            # No results found
            if query_language == 'th':
                response_text = "สวัสดีครับ! ขออภัย ไม่พบข้อมูลเพลงที่ตรงกับคำค้นหาของคุณในฐานข้อมูลของเรา โปรดลองค้นหาด้วยชื่อเพลงหรือชื่อศิลปินอื่น"
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