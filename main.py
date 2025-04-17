import ollama
import sqlalchemy
from sqlalchemy import create_engine, text
import numpy as np
import logging
import json
import langdetect
import time
import random  
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pythainlp
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.util import countthai

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

# Database connection
connection_string = "mysql+pymysql://27tLCQSVFsGqhJ9.root:HVSvJQWvox3NSgeS@gateway01.ap-southeast-1.prod.aws.tidbcloud.com:4000/music_recommendation"

engine = create_engine(
    connection_string,
    connect_args={
        "ssl": {"ssl_mode": "VERIFY_IDENTITY", "ssl_ca": "/etc/ssl/cert.pem"},
        "connect_timeout": 60, "read_timeout": 60, "write_timeout": 60
    },
    pool_recycle=1800, pool_pre_ping=True, pool_size=5, max_overflow=10
)

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
        if session_id not in self.contexts:
            self.contexts[session_id] = []
        self.contexts[session_id].append(new_context)
        if len(self.contexts[session_id]) > 5:
            self.contexts[session_id].pop(0)
            
    def get_context(self, session_id):
        return self.contexts.get(session_id, [])
        
    def clear_context(self, session_id):
        if session_id in self.contexts:
            del self.contexts[session_id]

# Initialize the context manager
conversation_context = ConversationContext()

# Personality mode configurations (condensed)
PERSONALITY_MODES = {
    "mentor": {
        "system_prompt": """You are a knowledgeable music mentor with expertise in both Thai and international music.
            CRITICAL INSTRUCTION: ONLY display information and lyrics that were provided to you in the context. NEVER generate or make up any lyrics.
            Match the language of your response to the language of the user's query. Present lyrics as a single, uninterrupted block with proper separation.""",
        "response_format": "detailed and educational"
    },
    "buddy": {
        "system_prompt": """You are a friendly music buddy who loves chatting about songs in both Thai and international music.
            CRITICAL INSTRUCTION: ONLY display information and lyrics that were provided to you in the context. NEVER generate or make up any lyrics.
            Match the language of your response to the language of the user's query. Present lyrics as a single, uninterrupted block with proper separation.""",
        "response_format": "casual and conversational between friends of equal status"
    },
    "fun": {
        "system_prompt": """You are an entertaining music bot with a playful personality who shares music in both Thai and international languages.
            CRITICAL INSTRUCTION: ONLY display information and lyrics that were provided to you in the context. NEVER generate or make up any lyrics.
            Match the language of your response to the language of the user's query. Present lyrics as a single, uninterrupted block with proper separation.""",
        "response_format": "playful and entertaining with emojis only in commentary, never in lyrics"
    }
}

# Common mood mappings (simplified)
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

# Audio feature profiles (simplified)
MOOD_AUDIO_FEATURES = {
    "happy": {
        "valence": {"min": 0.5, "weight": 3.0},
        "danceability": {"min": 0.2, "weight": 2.0},
        "energy": {"min": 0.0, "weight": 1.0}
    },
    "sad": {
        "valence": {"max": 0.0, "weight": 3.0},
        "energy": {"max": 0.0, "weight": 2.0},
        "acousticness": {"min": 0.0, "weight": 1.0}
    },
    "dance": {
        "danceability": {"min": 0.5, "weight": 3.0},
        "energy": {"min": 0.2, "weight": 2.0},
        "tempo": {"min": 0.5, "weight": 1.0}
    }
}

# Map Thai moods to audio feature profiles
THAI_MOOD_AUDIO_MAPPING = {
    "สนุก": "happy", "เศร้า": "sad", "รัก": "love", "เต้น": "dance"
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
            response = ollama.embeddings(model='llama3.1', prompt=question)
            if not response or 'embedding' not in response:
                logger.error(f"Invalid embedding response structure: {response}")
                continue
                return []
    
        except Exception as e:
            logger.error(f"Error in song search: {str(e)}")
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
    song_title = song_data.get('song_title', '')
    artist = song_data.get('artist', '')
    lyrics = song_data.get('processed_text', '')
    
    processed_lyrics = lyrics.replace('\\n', '\n').replace('\\r', '\r') if lyrics else ''
    
    if language == 'th':
        system_prompt = f"""คุณเป็นผู้เชี่ยวชาญในการวิเคราะห์เพลงและดนตรี
        วิเคราะห์ความหมาย แนวคิด และบริบทของเพลงโดยใช้เนื้อเพลงที่ได้รับ
        ห้ามแต่งเนื้อเพลงหรือเพิ่มข้อมูลที่ไม่มีในเนื้อเพลง
        ตอบในรูปแบบ {personality["response_format"]} ตามบุคลิกของคุณ
        """
        
        user_prompt = f"""คำถาม: "{question}"
        
        เพลง: {song_title}
        ศิลปิน: {artist}
        
        เนื้อเพลง:
        {processed_lyrics if processed_lyrics else '[ไม่พบเนื้อเพลงในฐานข้อมูล]'}
        
        กรุณาวิเคราะห์เพลงนี้ อธิบายความหมาย สาระสำคัญ และบริบทของเพลง
        """
    else:
        system_prompt = f"""You are an expert music analyst who provides insight into songs.
        Analyze the meaning, themes, and context of the song using only the provided lyrics.
        NEVER make up or add to the lyrics that aren't provided.
        Respond in a {personality["response_format"]} style according to your personality.
        """
        
        user_prompt = f"""Query: "{question}"
        
        Song: {song_title}
        Artist: {artist}
        
        Lyrics:
        {processed_lyrics if processed_lyrics else '[NO LYRICS FOUND IN DATABASE]'}
        
        Please analyze this song, explaining its meaning, themes, and context.
        """
    
    response_text = await generate_llm_response(user_prompt, system_prompt)
    return response_text

async def hybrid_search(question, query_intent, query_embedding=None, top_k=5):
    logger.info(f"Performing hybrid search for: '{question}'")
    
    intent = query_intent.get('intent', 'general_query')
    
    if query_embedding is None:
        query_embedding = await get_query_embedding(question)
    
    direct_results = []
    vector_results = []
    intent_results = []
    audio_feature_results = []
    
    try:
        # Special handling for lyrics search intent
        if query_intent.get('intent') == 'lyrics_search':
            search_phrase = query_intent.get('song_title', question)
            
            # First attempt: Direct phrase match
            phrase_results = await search_lyrics_by_phrase(search_phrase, top_k=1)
            
            if phrase_results:
                logger.info(f"Found direct phrase match in lyrics: '{phrase_results[0]['song_title']}'")
                return phrase_results
                
            # Second attempt: Vector search for lyrics
            if query_embedding:
                # Handle with vector search (implementation simplified)
                vector_results = await search_with_vector(query_embedding, top_k)
                if vector_results:
                    return vector_results
                    
            # Fallback to title matching
            if 'song_title' in query_intent and query_intent['song_title']:
                if 'artist' in query_intent and query_intent['artist']:
                    direct_results = await search_exact_song(query_intent['song_title'], query_intent['artist'])
                else:
                    direct_results = await search_exact_song(query_intent['song_title'])
                
                if direct_results:
                    return direct_results
        
        # For mood-based search, use audio features
        if intent == 'mood_search' and query_intent.get('mood'):
            mood = query_intent.get('mood')
            audio_feature_results = await search_by_audio_features(mood, top_k=top_k)
                
            if audio_feature_results:
                return audio_feature_results
        
        # Song title search
        if 'song_title' in query_intent and query_intent['song_title'] and intent != 'lyrics_search':
            if 'artist' in query_intent and query_intent['artist']:
                direct_results = await search_exact_song(query_intent['song_title'], query_intent['artist'])
            else:
                direct_results = await search_exact_song(query_intent['song_title'])
            
            if direct_results:
                return direct_results
                    
        # Artist recommendations handling
        if intent == 'artist_recommendations' and query_intent.get('artist'):
            artist = query_intent.get('artist')
            
            with engine.connect() as connection:
                # Find songs from this artist
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
                
                if intent_results:
                    return intent_results
        
        # Vector search as fallback
        if query_embedding:
            vector_results = await search_with_vector(query_embedding, top_k=top_k)
            if vector_results:
                return vector_results
            
        # Return any available results in priority order
        if audio_feature_results:
            return audio_feature_results
        if intent_results:
            return intent_results
        if direct_results:
            return direct_results
        if vector_results:
            return vector_results
            
        return []
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {str(e)}")
        return []



async def parse_query_with_llm(question, context=None, max_retries=3):
    is_thai = countthai(question) / len(question) > 0.15 if question else False
    
    # Extract context information
    context_str = ""
    previous_song = None
    if context and len(context) > 0:
        context_str = "Previous conversation:\n"
        for i, ctx in enumerate(context[-3:]):
            context_str += f"- Query {i+1}: '{ctx.get('query')}'\n"
            if 'ambiguous_song' in ctx:
                context_str += f"  (This was asking for clarification about song: {ctx.get('ambiguous_song')})\n"
                previous_song = ctx.get('ambiguous_song')
    
    # Check if this might be a follow-up
    follow_up_indicator = False
    if is_thai and previous_song:
        if "ของ" in question or "จาก" in question:
            tokens = word_tokenize(question, engine="newmm")
            request_tokens = [token for token in tokens if token.lower() in THAI_REQUEST_WORDS]
            if len(request_tokens) / len(tokens) > 0.3:
                follow_up_indicator = True
                logger.info(f"Detected follow-up query for song: {previous_song}")
    
    for attempt in range(max_retries):
        try:
            # Format system prompt
            system_prompt = """You are a query understanding assistant specialized in music queries in Thai and English.
            CRITICAL INSTRUCTIONS FOR RESPONSE FORMAT:
            - Your response MUST be VALID JSON only
            - Do not include ANY text before or after the JSON
            
            CRITICAL INSTRUCTIONS FOR THAI LANGUAGE:
            1. Always identify Thai phrases as REQUEST INDICATORS, not song titles
            2. Properly identify analysis requests
            3. When users ask for songs with emotions or moods, set intent to "mood_search"
            4. For Thai queries, be extremely careful to avoid including Thai request words in the song_title field.
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

                Extract the song title, artist, and intent from this music query.

                IMPORTANT: Respond with ONLY the JSON object. No explanations, no markdown, JUST the JSON.

                {{
                    "song_title": null or "extracted title",
                    "artist": null or "extracted artist",
                    "intent": "lyrics_search|song_info|artist_info|mood_search|artist_recommendations|continuation|general_query",
                    "mood": null or "sad|happy|love|etc",
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
            
            # Try to extract valid JSON
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
                
                # For Thai queries, double-check song_title doesn't contain request words
                if is_thai and parsed_data.get('song_title'):
                    for word in THAI_REQUEST_WORDS:
                        if word in parsed_data['song_title']:
                            parsed_data['song_title'] = parsed_data['song_title'].replace(word, "").strip()
                
                # Handle specific edge case for Thai artist clarification
                if is_thai and "ของ" in question and parsed_data.get('artist') and not parsed_data.get('song_title'):
                    if previous_song:
                        parsed_data['song_title'] = previous_song
                        parsed_data['intent'] = 'continuation'
                        parsed_data['refers_to_previous'] = True
                        logger.info(f"Identified as artist clarification for previous song: {previous_song}")
                
                return parsed_data
            else:
                logger.error("No valid JSON found in LLM response")
                raise ValueError("Invalid LLM response format")
                
        except Exception as e:
            logger.error(f"Error in LLM query parsing (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(sleep_time)
    
    # If LLM parsing fails, fall back to rule-based parsing
    logger.warning("Falling back to rule-based query parsing")
    return {
        'intent': 'general_query',  
        'entity': None,
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
    if not query_embedding:
        logger.error("Cannot perform search: No valid query embedding provided")
        return []
    
    vector_dim = len(query_embedding)
    
    try:
        with engine.connect() as connection:
            vector_json = json.dumps(query_embedding)
            
            query = text("""
            SELECT s.song_id, s.track_name, s.track_artist, 
                   s.original_lyrics, s.cleaned_lyrics,
                   1 - (v.lyrics_embedding <=> CAST(:vector_json AS VECTOR(:dim))) AS lyrics_similarity,
                   1 - (v.metadata_embedding <=> CAST(:vector_json AS VECTOR(:dim))) AS metadata_similarity
            FROM songs s
            JOIN song_embeddings_vector v ON s.song_id = v.song_id
            ORDER BY GREATEST(lyrics_similarity, metadata_similarity) DESC
            LIMIT :top_k
            """)
            
            results = connection.execute(query, {
                "vector_json": vector_json,
                "dim": vector_dim,
                "top_k": top_k
            }).fetchall()
            
            similarities = []
            for row in results:
                max_sim = max(row.lyrics_similarity or 0, row.metadata_similarity or 0)
                
                if max_sim > 0.99:
                    max_sim = 0.85 + (max_sim - 0.99) * 0.15
                
                if max_sim > 0.6:
                    similarities.append({
                        "id": row.song_id,
                        "song_title": row.track_name,
                        "artist": row.track_artist,
                        "processed_text": row.original_lyrics,
                        "cleaned_text": row.cleaned_lyrics,
                        "similarity": float(max_sim),
                        "match_type": "vector"
                    })
            
            return similarities
            
    except Exception as e:
        logger.error(f"Error in TiDB vector search: {str(e)}")
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
                    match_score = getattr(row, 'match_score', 0)
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
            
            songs = []
            for row in results:
                # Calculate a mood match score
                mood_score = 0.85  # Simplified from calculate_mood_score
                
                songs.append({
                    "id": row.song_id,
                    "song_title": row.track_name,
                    "artist": row.track_artist,
                    "processed_text": row.original_lyrics,
                    "cleaned_text": row.cleaned_lyrics,
                    "similarity": mood_score,
                    "match_type": "mood_feature",
                    "audio_features": {
                        "danceability": getattr(row, 'danceability', None),
                        "energy": getattr(row, 'energy', None),
                        "valence": getattr(row, 'valence', None),
                        "acousticness": getattr(row, 'acousticness', None),
                        "tempo": getattr(row, 'tempo', None)
                    }
                })
            
            return songs
            
    except Exception as e:
        logger.error(f"Error in audio feature search: {str(e)}")
        return []

import ollama
import sqlalchemy
from sqlalchemy import create_engine, text
import numpy as np
import logging
import json
import langdetect
import time
import random  
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pythainlp
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.util import countthai

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

# Database connection
connection_string = "mysql+pymysql://27tLCQSVFsGqhJ9.root:HVSvJQWvox3NSgeS@gateway01.ap-southeast-1.prod.aws.tidbcloud.com:4000/music_recommendation"

engine = create_engine(
    connection_string,
    connect_args={
        "ssl": {"ssl_mode": "VERIFY_IDENTITY", "ssl_ca": "/etc/ssl/cert.pem"},
        "connect_timeout": 60, "read_timeout": 60, "write_timeout": 60
    },
    pool_recycle=1800, pool_pre_ping=True, pool_size=5, max_overflow=10
)

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
        if session_id not in self.contexts:
            self.contexts[session_id] = []
        self.contexts[session_id].append(new_context)
        if len(self.contexts[session_id]) > 5:
            self.contexts[session_id].pop(0)
            
    def get_context(self, session_id):
        return self.contexts.get(session_id, [])
        
    def clear_context(self, session_id):
        if session_id in self.contexts:
            del self.contexts[session_id]

# Initialize the context manager
conversation_context = ConversationContext()

# Personality mode configurations (condensed)
PERSONALITY_MODES = {
    "mentor": {
        "system_prompt": """You are a knowledgeable music mentor with expertise in both Thai and international music.
            CRITICAL INSTRUCTION: ONLY display information and lyrics that were provided to you in the context. NEVER generate or make up any lyrics.
            Match the language of your response to the language of the user's query. Present lyrics as a single, uninterrupted block with proper separation.""",
        "response_format": "detailed and educational"
    },
    "buddy": {
        "system_prompt": """You are a friendly music buddy who loves chatting about songs in both Thai and international music.
            CRITICAL INSTRUCTION: ONLY display information and lyrics that were provided to you in the context. NEVER generate or make up any lyrics.
            Match the language of your response to the language of the user's query. Present lyrics as a single, uninterrupted block with proper separation.""",
        "response_format": "casual and conversational between friends of equal status"
    },
    "fun": {
        "system_prompt": """You are an entertaining music bot with a playful personality who shares music in both Thai and international languages.
            CRITICAL INSTRUCTION: ONLY display information and lyrics that were provided to you in the context. NEVER generate or make up any lyrics.
            Match the language of your response to the language of the user's query. Present lyrics as a single, uninterrupted block with proper separation.""",
        "response_format": "playful and entertaining with emojis only in commentary, never in lyrics"
    }
}

# Common mood mappings (simplified)
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

# Audio feature profiles (simplified)
MOOD_AUDIO_FEATURES = {
    "happy": {
        "valence": {"min": 0.5, "weight": 3.0},
        "danceability": {"min": 0.2, "weight": 2.0},
        "energy": {"min": 0.0, "weight": 1.0}
    },
    "sad": {
        "valence": {"max": 0.0, "weight": 3.0},
        "energy": {"max": 0.0, "weight": 2.0},
        "acousticness": {"min": 0.0, "weight": 1.0}
    },
    "dance": {
        "danceability": {"min": 0.5, "weight": 3.0},
        "energy": {"min": 0.2, "weight": 2.0},
        "tempo": {"min": 0.5, "weight": 1.0}
    }
}

# Map Thai moods to audio feature profiles
THAI_MOOD_AUDIO_MAPPING = {
    "สนุก": "happy", "เศร้า": "sad", "รัก": "love", "เต้น": "dance"
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
            response = ollama.embeddings(model='llama3.1', prompt=question)
            if not response or 'embedding' not in response:
                logger.error(f"Invalid embedding response structure: {response}")
                continue
                return []
    
        except Exception as e:
            logger.error(f"Error in song search: {str(e)}")
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
    song_title = song_data.get('song_title', '')
    artist = song_data.get('artist', '')
    lyrics = song_data.get('processed_text', '')
    
    processed_lyrics = lyrics.replace('\\n', '\n').replace('\\r', '\r') if lyrics else ''
    
    if language == 'th':
        system_prompt = f"""คุณเป็นผู้เชี่ยวชาญในการวิเคราะห์เพลงและดนตรี
        วิเคราะห์ความหมาย แนวคิด และบริบทของเพลงโดยใช้เนื้อเพลงที่ได้รับ
        ห้ามแต่งเนื้อเพลงหรือเพิ่มข้อมูลที่ไม่มีในเนื้อเพลง
        ตอบในรูปแบบ {personality["response_format"]} ตามบุคลิกของคุณ
        """
        
        user_prompt = f"""คำถาม: "{question}"
        
        เพลง: {song_title}
        ศิลปิน: {artist}
        
        เนื้อเพลง:
        {processed_lyrics if processed_lyrics else '[ไม่พบเนื้อเพลงในฐานข้อมูล]'}
        
        กรุณาวิเคราะห์เพลงนี้ อธิบายความหมาย สาระสำคัญ และบริบทของเพลง
        """
    else:
        system_prompt = f"""You are an expert music analyst who provides insight into songs.
        Analyze the meaning, themes, and context of the song using only the provided lyrics.
        NEVER make up or add to the lyrics that aren't provided.
        Respond in a {personality["response_format"]} style according to your personality.
        """
        
        user_prompt = f"""Query: "{question}"
        
        Song: {song_title}
        Artist: {artist}
        
        Lyrics:
        {processed_lyrics if processed_lyrics else '[NO LYRICS FOUND IN DATABASE]'}
        
        Please analyze this song, explaining its meaning, themes, and context.
        """
    
    response_text = await generate_llm_response(user_prompt, system_prompt)
    return response_text

async def hybrid_search(question, query_intent, query_embedding=None, top_k=5):
    logger.info(f"Performing hybrid search for: '{question}'")
    
    intent = query_intent.get('intent', 'general_query')
    
    if query_embedding is None:
        query_embedding = await get_query_embedding(question)
    
    direct_results = []
    vector_results = []
    intent_results = []
    audio_feature_results = []
    
    try:
        # Special handling for lyrics search intent
        if query_intent.get('intent') == 'lyrics_search':
            search_phrase = query_intent.get('song_title', question)
            
            # First attempt: Direct phrase match
            phrase_results = await search_lyrics_by_phrase(search_phrase, top_k=1)
            
            if phrase_results:
                logger.info(f"Found direct phrase match in lyrics: '{phrase_results[0]['song_title']}'")
                return phrase_results
                
            # Second attempt: Vector search for lyrics
            if query_embedding:
                # Handle with vector search (implementation simplified)
                vector_results = await search_with_vector(query_embedding, top_k)
                if vector_results:
                    return vector_results
                    
            # Fallback to title matching
            if 'song_title' in query_intent and query_intent['song_title']:
                if 'artist' in query_intent and query_intent['artist']:
                    direct_results = await search_exact_song(query_intent['song_title'], query_intent['artist'])
                else:
                    direct_results = await search_exact_song(query_intent['song_title'])
                
                if direct_results:
                    return direct_results
        
        # For mood-based search, use audio features
        if intent == 'mood_search' and query_intent.get('mood'):
            mood = query_intent.get('mood')
            audio_feature_results = await search_by_audio_features(mood, top_k=top_k)
                
            if audio_feature_results:
                return audio_feature_results
        
        # Song title search
        if 'song_title' in query_intent and query_intent['song_title'] and intent != 'lyrics_search':
            if 'artist' in query_intent and query_intent['artist']:
                direct_results = await search_exact_song(query_intent['song_title'], query_intent['artist'])
            else:
                direct_results = await search_exact_song(query_intent['song_title'])
            
            if direct_results:
                return direct_results
                    
        # Artist recommendations handling
        if intent == 'artist_recommendations' and query_intent.get('artist'):
            artist = query_intent.get('artist')
            
            with engine.connect() as connection:
                # Find songs from this artist
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
                
                if intent_results:
                    return intent_results
        
        # Vector search as fallback
        if query_embedding:
            vector_results = await search_with_vector(query_embedding, top_k=top_k)
            if vector_results:
                return vector_results
            
        # Return any available results in priority order
        if audio_feature_results:
            return audio_feature_results
        if intent_results:
            return intent_results
        if direct_results:
            return direct_results
        if vector_results:
            return vector_results
            
        return []
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {str(e)}")
        return []

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
        
        logger.info(f"=== NEW REQUEST === '{question}' in {mode} mode")
        
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
        
        # 2. Handle artist recommendations intent directly
        if intent_data.get('intent') == 'artist_recommendations' and intent_data.get('artist'):
            artist = intent_data.get('artist')
            query_language = intent_data.get('detected_language', 'en')
            
            result = await handle_artist_recommendations(artist, mode, query_language, limit=5)
            result["processing_time"] = round(time.time() - start_time, 2)
            
            return result
        
        # 3. Perform search based on intent
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

        # 5. Create context from results for LLM
        context_items = []
        sorted_results = sorted(similar_results, key=lambda x: x.get('similarity', 0), reverse=True)
        
        for item in sorted_results:
            processed_text = item.get('processed_text', '').replace('\\n', '\n').replace('\\r', '\r')
            
            if item == sorted_results[0]:
                context_items.append(
                    f"BEST MATCH: Song: {item['song_title']} by {item['artist']}\n"
                    f"(Match type: {item.get('match_type', 'unknown')})\n"
                    f"Lyrics: {processed_text if processed_text.strip() else '[NO LYRICS FOUND IN DATABASE]'}"
                )
            else:
                context_items.append(
                    f"Song: {item['song_title']} by {item['artist']}\n"
                    f"Lyrics: {processed_text if processed_text.strip() else '[NO LYRICS FOUND IN DATABASE]'}"
                )
        
        context = "\n\n".join(context_items)
        
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
                        
                        return {
                            "response": response_text,
                            "mode": mode,
                            "intent": "clarification",
                            "sources": [{"title": item["song_title"], "artist": item["artist"]} for item in similar_results],
                            "processing_time": round(time.time() - start_time, 2)
                        }
            
            # Normal response generation
            if query_language == 'th':
                prompt = f"""
                คำถามของผู้ใช้: "{question}"
                
                ข้อมูลเพลงที่ค้นพบในฐานข้อมูล:
                {context}
                
                กรุณาตอบเป็นภาษาไทยในรูปแบบ {personality["response_format"]} ตามบุคลิกของคุณ
                แสดงเฉพาะข้อมูลและเนื้อเพลงที่ระบุไว้ข้างต้นเท่านั้น
                """
            else:
                prompt = f"""
                User query: "{question}"
                
                Songs information found in database:
                {context}
                
                Please respond in English in a {personality["response_format"]} style.
                ONLY display information and lyrics shown above.
                """
            
            response_text = await generate_llm_response(prompt, personality["system_prompt"])
        else:
            # No results found
            if query_language == 'th':
                response_text = "ขออภัย ไม่พบข้อมูลเพลงที่ตรงกับคำค้นหาของคุณในฐานข้อมูลของเรา"
            else:
                response_text = "Sorry, I couldn't find any songs matching your query in our database."
                    
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
     


async def parse_query_with_llm(question, context=None, max_retries=3):
    is_thai = countthai(question) / len(question) > 0.15 if question else False
    
    # Extract context information
    context_str = ""
    previous_song = None
    if context and len(context) > 0:
        context_str = "Previous conversation:\n"
        for i, ctx in enumerate(context[-3:]):
            context_str += f"- Query {i+1}: '{ctx.get('query')}'\n"
            if 'ambiguous_song' in ctx:
                context_str += f"  (This was asking for clarification about song: {ctx.get('ambiguous_song')})\n"
                previous_song = ctx.get('ambiguous_song')
    
    # Check if this might be a follow-up
    follow_up_indicator = False
    if is_thai and previous_song:
        if "ของ" in question or "จาก" in question:
            tokens = word_tokenize(question, engine="newmm")
            request_tokens = [token for token in tokens if token.lower() in THAI_REQUEST_WORDS]
            if len(request_tokens) / len(tokens) > 0.3:
                follow_up_indicator = True
                logger.info(f"Detected follow-up query for song: {previous_song}")
    
    for attempt in range(max_retries):
        try:
            # Format system prompt
            system_prompt = """You are a query understanding assistant specialized in music queries in Thai and English.
            CRITICAL INSTRUCTIONS FOR RESPONSE FORMAT:
            - Your response MUST be VALID JSON only
            - Do not include ANY text before or after the JSON
            
            CRITICAL INSTRUCTIONS FOR THAI LANGUAGE:
            1. Always identify Thai phrases as REQUEST INDICATORS, not song titles
            2. Properly identify analysis requests
            3. When users ask for songs with emotions or moods, set intent to "mood_search"
            4. For Thai queries, be extremely careful to avoid including Thai request words in the song_title field.
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

                Extract the song title, artist, and intent from this music query.

                IMPORTANT: Respond with ONLY the JSON object. No explanations, no markdown, JUST the JSON.

                {{
                    "song_title": null or "extracted title",
                    "artist": null or "extracted artist",
                    "intent": "lyrics_search|song_info|artist_info|mood_search|artist_recommendations|continuation|general_query",
                    "mood": null or "sad|happy|love|etc",
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
            
            # Try to extract valid JSON
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
                
                # For Thai queries, double-check song_title doesn't contain request words
                if is_thai and parsed_data.get('song_title'):
                    for word in THAI_REQUEST_WORDS:
                        if word in parsed_data['song_title']:
                            parsed_data['song_title'] = parsed_data['song_title'].replace(word, "").strip()
                
                # Handle specific edge case for Thai artist clarification
                if is_thai and "ของ" in question and parsed_data.get('artist') and not parsed_data.get('song_title'):
                    if previous_song:
                        parsed_data['song_title'] = previous_song
                        parsed_data['intent'] = 'continuation'
                        parsed_data['refers_to_previous'] = True
                        logger.info(f"Identified as artist clarification for previous song: {previous_song}")
                
                return parsed_data
            else:
                logger.error("No valid JSON found in LLM response")
                raise ValueError("Invalid LLM response format")
                
        except Exception as e:
            logger.error(f"Error in LLM query parsing (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(sleep_time)
    
    # If LLM parsing fails, fall back to rule-based parsing
    logger.warning("Falling back to rule-based query parsing")
    return {
        'intent': 'general_query',  
        'entity': None,
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
    if not query_embedding:
        logger.error("Cannot perform search: No valid query embedding provided")
        return []
    
    vector_dim = len(query_embedding)
    
    try:
        with engine.connect() as connection:
            vector_json = json.dumps(query_embedding)
            
            query = text("""
            SELECT s.song_id, s.track_name, s.track_artist, 
                   s.original_lyrics, s.cleaned_lyrics,
                   1 - (v.lyrics_embedding <=> CAST(:vector_json AS VECTOR(:dim))) AS lyrics_similarity,
                   1 - (v.metadata_embedding <=> CAST(:vector_json AS VECTOR(:dim))) AS metadata_similarity
            FROM songs s
            JOIN song_embeddings_vector v ON s.song_id = v.song_id
            ORDER BY GREATEST(lyrics_similarity, metadata_similarity) DESC
            LIMIT :top_k
            """)
            
            results = connection.execute(query, {
                "vector_json": vector_json,
                "dim": vector_dim,
                "top_k": top_k
            }).fetchall()
            
            similarities = []
            for row in results:
                max_sim = max(row.lyrics_similarity or 0, row.metadata_similarity or 0)
                
                if max_sim > 0.99:
                    max_sim = 0.85 + (max_sim - 0.99) * 0.15
                
                if max_sim > 0.6:
                    similarities.append({
                        "id": row.song_id,
                        "song_title": row.track_name,
                        "artist": row.track_artist,
                        "processed_text": row.original_lyrics,
                        "cleaned_text": row.cleaned_lyrics,
                        "similarity": float(max_sim),
                        "match_type": "vector"
                    })
            
            return similarities
            
    except Exception as e:
        logger.error(f"Error in TiDB vector search: {str(e)}")
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
                    match_score = getattr(row, 'match_score', 0)
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
            
            songs = []
            for row in results:
                # Calculate a mood match score
                mood_score = 0.85  # Simplified from calculate_mood_score
                
                songs.append({
                    "id": row.song_id,
                    "song_title": row.track_name,
                    "artist": row.track_artist,
                    "processed_text": row.original_lyrics,
                    "cleaned_text": row.cleaned_lyrics,
                    "similarity": mood_score,
                    "match_type": "mood_feature",
                    "audio_features": {
                        "danceability": getattr(row, 'danceability', None),
                        "energy": getattr(row, 'energy', None),
                        "valence": getattr(row, 'valence', None),
                        "acousticness": getattr(row, 'acousticness', None),
                        "tempo": getattr(row, 'tempo', None)
                    }
                })
            
            return songs
            
    except Exception as e:
        logger.error(f"Error in audio feature search: {str(e)}")
        return []


async def search_exact_song(song_title, artist=None):
    try:
        with engine.connect() as connection:
            if artist:
                # Title + artist search with exact match
                query = text("""
                SELECT song_id, track_name, track_artist, original_lyrics, cleaned_lyrics 
                FROM songs 
                WHERE LOWER(track_name) = LOWER(:exact_title) 
                  AND LOWER(track_artist) LIKE LOWER(:artist_pattern)
                LIMIT 1
                """)
                
                results = connection.execute(query, {
                    "exact_title": song_title.lower(),
                    "artist_pattern": f"%{artist.lower()}%"
                }).fetchall()
            else:
                # Title-only search with exact match
                query = text("""
                SELECT song_id, track_name, track_artist, original_lyrics, cleaned_lyrics 
                FROM songs 
                WHERE LOWER(track_name) = LOWER(:exact_title)
                ORDER BY CASE WHEN popularity_score IS NULL THEN 0 ELSE popularity_score END DESC
                LIMIT 5
                """)
                
                results = connection.execute(query, {
                    "exact_title": song_title.lower()
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
                LIMIT 5
                """)
                
                all_params = {**word_params}
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
                        "similarity": getattr(row, 'match_score', 0.7),
                        "match_type": "fuzzy"
                    } for row in fuzzy_results]
            
            return []
    
    except Exception as e:
        logger.error(f"Error in song search: {str(e)}")
        return []
    


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
        
        logger.info(f"=== NEW REQUEST === '{question}' in {mode} mode")
        
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
        
        # 2. Handle artist recommendations intent directly
        if intent_data.get('intent') == 'artist_recommendations' and intent_data.get('artist'):
            artist = intent_data.get('artist')
            query_language = intent_data.get('detected_language', 'en')
            
            result = await handle_artist_recommendations(artist, mode, query_language, limit=5)
            result["processing_time"] = round(time.time() - start_time, 2)
            
            return result
        
        # 3. Perform search based on intent
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

        # 5. Create context from results for LLM
        context_items = []
        sorted_results = sorted(similar_results, key=lambda x: x.get('similarity', 0), reverse=True)
        
        for item in sorted_results:
            processed_text = item.get('processed_text', '').replace('\\n', '\n').replace('\\r', '\r')
            
            if item == sorted_results[0]:
                context_items.append(
                    f"BEST MATCH: Song: {item['song_title']} by {item['artist']}\n"
                    f"(Match type: {item.get('match_type', 'unknown')})\n"
                    f"Lyrics: {processed_text if processed_text.strip() else '[NO LYRICS FOUND IN DATABASE]'}"
                )
            else:
                context_items.append(
                    f"Song: {item['song_title']} by {item['artist']}\n"
                    f"Lyrics: {processed_text if processed_text.strip() else '[NO LYRICS FOUND IN DATABASE]'}"
                )
        
        context = "\n\n".join(context_items)
        
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
                        
                        return {
                            "response": response_text,
                            "mode": mode,
                            "intent": "clarification",
                            "sources": [{"title": item["song_title"], "artist": item["artist"]} for item in similar_results],
                            "processing_time": round(time.time() - start_time, 2)
                        }
            
            # Normal response generation
            if query_language == 'th':
                prompt = f"""
                คำถามของผู้ใช้: "{question}"
                
                ข้อมูลเพลงที่ค้นพบในฐานข้อมูล:
                {context}
                
                กรุณาตอบเป็นภาษาไทยในรูปแบบ {personality["response_format"]} ตามบุคลิกของคุณ
                แสดงเฉพาะข้อมูลและเนื้อเพลงที่ระบุไว้ข้างต้นเท่านั้น
                """
            else:
                prompt = f"""
                User query: "{question}"
                
                Songs information found in database:
                {context}
                
                Please respond in English in a {personality["response_format"]} style.
                ONLY display information and lyrics shown above.
                """
            
            response_text = await generate_llm_response(prompt, personality["system_prompt"])
        else:
            # No results found
            if query_language == 'th':
                response_text = "ขออภัย ไม่พบข้อมูลเพลงที่ตรงกับคำค้นหาของคุณในฐานข้อมูลของเรา"
            else:
                response_text = "Sorry, I couldn't find any songs matching your query in our database."
                    
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
 