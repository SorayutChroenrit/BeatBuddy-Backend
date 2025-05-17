# app/music.py
import time
import json
import logging
import random
import re
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, text, case, or_, and_, desc
from groq import Groq

from app.models import Song, ChatHistory
from app.schemas import AskQuestionResponse, SongInfo, ChatMessage
from app.config import settings

logger = logging.getLogger(__name__)

# Initialize Groq client
groq_client = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None

# Personality modes
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

class ConversationContext:
    def __init__(self):
        self.contexts = {}
        
    def update_context(self, session_id, new_context):
        if session_id not in self.contexts:
            self.contexts[session_id] = []
        self.contexts[session_id].append(new_context)
        if len(self.contexts[session_id]) > 5:
            self.contexts[session_id].pop(0)
            
    def get_context(self, session_id):
        return self.contexts.get(session_id, [])

# Global conversation context
conversation_context = ConversationContext()

class MusicService:
    def __init__(self, db: Session):
        self.db = db
    
    def detect_language(self, text: str) -> str:
        """Simple language detection"""
        # Count Thai characters
        thai_chars = len(re.findall(r'[\u0E00-\u0E7F]', text))
        if thai_chars / len(text) > 0.15:
            return 'th'
        return 'en'
    
    async def is_music_related_query(self, question: str) -> bool:
        """Check if query is music-related using Groq"""
        if not groq_client:
            return True  # Default to True if no Groq
        
        try:
            is_thai = self.detect_language(question) == 'th'
            
            if is_thai:
                system_prompt = """คุณเป็นตัวจำแนกคำถามที่เชี่ยวชาญด้านการระบุว่าคำถามนั้นเกี่ยวข้องกับเพลง เนื้อเพลง หรือศิลปินหรือไม่

ตอบ "ใช่" หาก คำถามเกี่ยวกับ:
- เพลง, ชื่อเพลง, เพลงใหม่, เพลงเก่า
- เนื้อเพลง, ท่อนเพลง, เนื้อร้อง
- ศิลปิน, นักร้อง, วงดนตรี, ผู้แต่งเพลง
- อัลบั้ม, มิวสิกวิดีโอ, คอนเสิร์ต
- แนวเพลง, อารมณ์เพลง (เศร้า, สนุก, เต้น)
- ความหมายเพลง, การวิเคราะห์เพลง

ตอบ "ไม่ใช่" หาก คำถามเกี่ยวกับ:  
- อาหาร, สภาพอากาศ, ข่าวสาร, เทคโนโลยี
- อื่นๆ ที่ไม่เกี่ยวข้องกับเพลง

ตอบเฉพาะ "ใช่" หรือ "ไม่ใช่" เท่านั้น"""
                user_prompt = f"คำถาม: \"{question}\"\n\nคำถามนี้เกี่ยวข้องกับเพลง เนื้อเพลง หรือศิลปินหรือไม่?"
            else:
                system_prompt = """You are an expert query classifier that determines if a question is related to songs, lyrics, or artists.

Answer "YES" if the question is about:
- Songs, song titles, new songs, old songs
- Lyrics, song verses, lyric lines
- Artists, singers, bands, songwriters
- Albums, music videos, concerts
- Music genres, song moods
- Song meanings, song analysis

Answer "NO" if the question is about:
- Food, weather, news, technology
- Other non-music topics

Answer ONLY "YES" or "NO" - no explanations."""
                user_prompt = f"Question: \"{question}\"\n\nIs this question related to songs, lyrics, or artists?"
            
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=10
            )
            
            answer = chat_completion.choices[0].message.content.strip()
            
            if is_thai:
                positive_answers = ["ใช่", "ใช้", "ใข่", "ใช", "yes", "เกี่ยวข้อง"]
                negative_answers = ["ไม่ใช่", "ไม่ใข่", "ไม่ใช้", "ไม่", "no", "ไม่เกี่ยวข้อง"]
                answer_lower = answer.lower()
                if any(neg in answer_lower for neg in negative_answers):
                    return False
                if any(pos in answer_lower for pos in positive_answers):
                    return True
            else:
                if "YES" in answer.upper():
                    return True
                elif "NO" in answer.upper():
                    return False
            
            return False
        except Exception as e:
            logger.error(f"Error in music query classification: {e}")
            return False
    
    async def parse_query_intent(self, question: str, context: list = None) -> Dict[str, Any]:
        """Parse query to extract intent, song title, artist, etc."""
        if not groq_client:
            # Fallback intent detection
            return {
                'intent': 'general_query',
                'song_title': None,
                'artist': None,
                'detected_language': self.detect_language(question),
                'original_query': question,
                'top_k': 5
            }
        
        is_thai = self.detect_language(question) == 'th'
        context_str = ""
        if context:
            context_str = "Previous conversation:\n"
            for i, ctx in enumerate(context[-3:]):
                context_str += f"- Query {i+1}: '{ctx.get('query')}'\n"
        
        system_prompt = """You are a query understanding assistant specialized in music queries in Thai and English.
        CRITICAL INSTRUCTIONS FOR RESPONSE FORMAT:
        - Your response MUST be VALID JSON only
        - Do not include ANY text before or after the JSON
        
        CRITICAL INSTRUCTIONS FOR INTENT DETECTION:
        1. For lyrics display requests like "show me lyrics of Song X", set intent to "lyrics_display"
        2. For lyrics identification queries like "what song has these lyrics", set intent to "lyrics_search"
        3. For artist recommendations, set intent to "artist_recommendations"
        4. For mood-based searches, set intent to "mood_search"
        5. Default to "general_query" when unsure"""
        
        user_prompt = f"""
        {context_str}
        Current query: "{question}"

        Based on the query, determine the user's intent and extract relevant information.
        
        IMPORTANT: Respond with ONLY the JSON object. No explanations, no markdown, JUST the JSON.

        {{
            "song_title": null or "extracted title",
            "artist": null or "extracted artist",
            "intent": "lyrics_search|lyrics_display|song_info|artist_info|mood_search|artist_recommendations|general_query",
            "mood": null or "sad|happy|love|dance|etc",
            "lyrics_fragment": null or "extracted lyrics if this is a lyrics identification query"
        }}
        """
        
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=500
            )
            
            content = chat_completion.choices[0].message.content.strip()
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                parsed_data = json.loads(json_content)
                parsed_data['detected_language'] = 'th' if is_thai else 'en'
                parsed_data['original_query'] = question
                parsed_data['top_k'] = 5
                return parsed_data
        except Exception as e:
            logger.error(f"Error in query parsing: {e}")
        
        # Fallback parsing
        return {
            'intent': 'general_query',
            'song_title': None,
            'artist': None,
            'detected_language': 'th' if is_thai else 'en',
            'original_query': question,
            'top_k': 5
        }
    
    async def search_songs(self, query: str, limit: int = 5, context_artist: str = None) -> List[SongInfo]:
        """
        Search for songs with improved exact match prioritization, context awareness, and better logging
        
        Args:
            query: The search query string
            limit: Maximum number of results to return
            context_artist: Optional artist from conversation context to prioritize in results
        """
        logger.info(f"Searching for songs with query: '{query}'")
        
        # Split query into potential song title and artist components
        query_parts = query.split(' by ')
        potential_title = query_parts[0].strip()
        potential_artist = query_parts[1].strip() if len(query_parts) > 1 else None
        
        logger.info(f"Potential title: '{potential_title}', potential artist: '{potential_artist}'")
        logger.info(f"Context artist: '{context_artist}'")
        
        try:
            # Import the SQLAlchemy func for database operations
            from sqlalchemy import func, or_, case, desc
            
            exact_title_matches = []
            partial_title_matches = []
            context_matches = []
            
            # STEP 1: Try exact title match first with specified artist if provided
            if potential_title:
                if potential_artist:
                    # When artist is explicitly specified in query
                    exact_query = self.db.query(Song).filter(
                        func.lower(Song.track_name) == func.lower(potential_title),
                        func.lower(Song.track_artist).like(f"%{potential_artist.lower()}%")
                    )
                    
                    exact_matches = exact_query.order_by(
                        case((Song.popularity_score == None, 0), else_=Song.popularity_score).desc()
                    ).limit(limit).all()
                    
                    if exact_matches:
                        logger.info(f"Found {len(exact_matches)} exact title+artist matches")
                        exact_title_matches = [
                            SongInfo(
                                id=song.song_id,
                                song_title=song.track_name or "",
                                artist=song.track_artist or "",
                                processed_text=song.original_lyrics,
                                similarity=1.0,
                                match_type="exact_title_artist"
                            )
                            for song in exact_matches
                        ]
                
                # STEP 2: Check for context artist match if no artist specified in query
                if not potential_artist and context_artist and not exact_title_matches:
                    # Try to find matches that use the artist from conversation context
                    context_query = self.db.query(Song).filter(
                        func.lower(Song.track_name) == func.lower(potential_title),
                        func.lower(Song.track_artist).like(f"%{context_artist.lower()}%")
                    )
                    
                    context_matches_raw = context_query.order_by(
                        case((Song.popularity_score == None, 0), else_=Song.popularity_score).desc()
                    ).limit(limit).all()
                    
                    if context_matches_raw:
                        logger.info(f"Found {len(context_matches_raw)} matches using context artist '{context_artist}'")
                        context_matches = [
                            SongInfo(
                                id=song.song_id,
                                song_title=song.track_name or "",
                                artist=song.track_artist or "",
                                processed_text=song.original_lyrics,
                                similarity=0.95,  # Slightly lower than exact match
                                match_type="context_artist"
                            )
                            for song in context_matches_raw
                        ]
                
                # STEP 3: If no artist-specific match, try title-only match
                if not exact_title_matches and not context_matches:
                    exact_title_query = self.db.query(Song).filter(
                        func.lower(Song.track_name) == func.lower(potential_title)
                    )
                    
                    exact_title_results = exact_title_query.order_by(
                        case((Song.popularity_score == None, 0), else_=Song.popularity_score).desc()
                    ).limit(limit * 2).all()  # Get more for disambiguation
                    
                    if exact_title_results:
                        logger.info(f"Found {len(exact_title_results)} exact title matches (without artist)")
                        exact_title_matches = [
                            SongInfo(
                                id=song.song_id,
                                song_title=song.track_name or "",
                                artist=song.track_artist or "",
                                processed_text=song.original_lyrics,
                                similarity=0.9,  # Lower than artist-specific match
                                match_type="exact_title_only"
                            )
                            for song in exact_title_results
                        ]
                        
                        # Check if we have multiple songs with same title but different artists
                        if len(exact_title_matches) > 1:
                            title_artists = {}
                            for match in exact_title_matches:
                                title_lower = match.song_title.lower()
                                if title_lower not in title_artists:
                                    title_artists[title_lower] = []
                                title_artists[title_lower].append(match.artist.lower())
                                
                            # If we have duplicate titles with different artists, mark them
                            for match in exact_title_matches:
                                if len(title_artists.get(match.song_title.lower(), [])) > 1:
                                    match.has_duplicates = True
                                    
                            logger.info(f"Title '{potential_title}' has duplicate versions by different artists")
            
            # STEP 4: Try partial title match if no exact matches
            if not exact_title_matches and not context_matches:
                logger.info("No exact matches, trying partial matches")
                partial_query = self.db.query(Song).filter(
                    func.lower(Song.track_name).like(f"%{potential_title.lower()}%")
                )
                
                if potential_artist:
                    partial_query = partial_query.filter(
                        func.lower(Song.track_artist).like(f"%{potential_artist.lower()}%")
                    )
                    
                partial_matches = partial_query.order_by(
                    case((Song.popularity_score == None, 0), else_=Song.popularity_score).desc()
                ).limit(limit).all()
                
                if partial_matches:
                    logger.info(f"Found {len(partial_matches)} partial title matches")
                    partial_title_matches = [
                        SongInfo(
                            id=song.song_id,
                            song_title=song.track_name or "",
                            artist=song.track_artist or "",
                            processed_text=song.original_lyrics,
                            similarity=0.8,
                            match_type="partial_title_match"
                        )
                        for song in partial_matches
                    ]
            
            # STEP 5: Combine and prioritize results
            all_matches = []
            
            # Prioritize based on match quality
            if exact_title_matches and potential_artist:
                # If user specified an artist, prioritize those exact matches
                all_matches.extend(exact_title_matches)
            elif context_matches:
                # Otherwise, if we have context matches, prioritize those
                all_matches.extend(context_matches)
                # Add any exact title matches that aren't already included
                all_matches.extend([m for m in exact_title_matches if all(m.id != cm.id for cm in context_matches)])
            else:
                # Otherwise just add exact title matches
                all_matches.extend(exact_title_matches)
            
            # Add partial matches that aren't duplicates of existing matches
            all_matches.extend([m for m in partial_title_matches if all(m.id != em.id for em in all_matches)])
            
            if all_matches:
                return all_matches[:limit]
            
            # STEP 6: Content-based search if all else fails
            content_conditions = []
            search_terms = query.lower().split()
            
            for term in search_terms:
                if len(term) < 2:  # Skip very short terms
                    continue
                    
                term_pattern = f"%{term}%"
                content_conditions.extend([
                    func.lower(Song.track_name).like(term_pattern),
                    func.lower(Song.track_artist).like(term_pattern),
                    func.lower(Song.original_lyrics).like(term_pattern)
                ])
            
            if content_conditions:
                content_matches = self.db.query(Song).filter(or_(*content_conditions)).order_by(
                    case((Song.popularity_score == None, 0), else_=Song.popularity_score).desc()
                ).limit(limit).all()
                
                if content_matches:
                    logger.info(f"Found {len(content_matches)} content-based matches")
                    content_based_matches = [
                        SongInfo(
                            id=song.song_id,
                            song_title=song.track_name or "",
                            artist=song.track_artist or "",
                            processed_text=song.original_lyrics,
                            similarity=0.6,
                            match_type="content_match"
                        )
                        for song in content_matches
                    ]
                    
                    return content_based_matches
            
            # STEP 7: Last resort - return some popular songs if nothing else matched
            logger.warning(f"No matches found for query: '{query}', returning fallback popular songs")
            fallback_songs = self.db.query(Song).order_by(
                case((Song.popularity_score == None, 0), else_=Song.popularity_score).desc()
            ).limit(limit).all()
            
            return [
                SongInfo(
                    id=song.song_id,
                    song_title=song.track_name or "",
                    artist=song.track_artist or "",
                    processed_text=song.original_lyrics,
                    similarity=0.4,  # Lower score for fallback matches
                    match_type="fallback"
                )
                for song in fallback_songs
            ]
        except Exception as e:
            logger.error(f"Error in song search: {str(e)}", exc_info=True)
            return []
    
    async def search_by_lyrics(self, lyrics_fragment: str, limit: int = 5) -> List[SongInfo]:
        """Enhanced search for songs by lyrics fragment with improved matching"""
        try:
            logger.info(f"Searching for lyrics: '{lyrics_fragment}'")
            
            # Normalize the lyrics fragment
            normalized_fragment = re.sub(r'\s+', ' ', lyrics_fragment.lower()).strip()
            
            with self.db.connection() as connection:
                # First try exact phrase matching
                exact_query = text("""
                SELECT song_id, track_name, track_artist, original_lyrics, cleaned_lyrics 
                FROM songs 
                WHERE LOWER(original_lyrics) LIKE :phrase_pattern
                OR LOWER(cleaned_lyrics) LIKE :phrase_pattern
                ORDER BY 
                    CASE 
                        WHEN LOWER(original_lyrics) LIKE :exact_pattern THEN 3
                        WHEN LOWER(cleaned_lyrics) LIKE :exact_pattern THEN 3
                        ELSE 1
                    END DESC,
                    CASE WHEN popularity_score IS NULL THEN 0 ELSE popularity_score END DESC
                LIMIT :limit
                """)
                
                results = connection.execute(exact_query, {
                    "phrase_pattern": f"%{normalized_fragment}%",
                    "exact_pattern": f"%{normalized_fragment}%",
                    "limit": limit
                }).fetchall()
                
                songs = []
                for row in results:
                    score = 0.9
                    if row.original_lyrics and normalized_fragment in row.original_lyrics.lower():
                        score = 0.98
                    elif row.cleaned_lyrics and normalized_fragment in row.cleaned_lyrics.lower():
                        score = 0.95
                    
                    songs.append(SongInfo(
                        id=row.song_id,
                        song_title=row.track_name,
                        artist=row.track_artist,
                        processed_text=row.original_lyrics,
                        similarity=score,
                        match_type="lyrics_exact"
                    ))
                
                # If no exact matches found, try word-based matching
                if not songs:
                    logger.info("No exact lyrics matches, trying word-based matching")
                    words = [w for w in normalized_fragment.split() if len(w) > 2]
                    
                    if words:
                        # Create a weighted word search that counts matches
                        word_conditions = []
                        word_params = {}
                        
                        for i, word in enumerate(words):
                            word_conditions.append(f"CASE WHEN LOWER(original_lyrics) LIKE :word_{i} THEN 1 ELSE 0 END")
                            word_params[f"word_{i}"] = f"%{word}%"
                        
                        word_query = text(f"""
                        SELECT song_id, track_name, track_artist, original_lyrics, cleaned_lyrics,
                            (
                                {" + ".join(word_conditions)}
                            ) / {len(words)}::float AS match_score
                        FROM songs
                        WHERE {" OR ".join([f"LOWER(original_lyrics) LIKE :word_{i}" for i in range(len(words))])}
                        ORDER BY match_score DESC, 
                            CASE WHEN popularity_score IS NULL THEN 0 ELSE popularity_score END DESC
                        LIMIT :limit
                        """)
                        
                        all_params = {**word_params, "limit": limit}
                        word_results = connection.execute(word_query, all_params).fetchall()
                        
                        logger.info(f"Word-based matches found: {len(word_results)}")
                        
                        for row in word_results:
                            match_score = float(getattr(row, 'match_score', 0))
                            if match_score > 0.2:  # Match at least 20% of words
                                songs.append(SongInfo(
                                    id=row.song_id,
                                    song_title=row.track_name,
                                    artist=row.track_artist,
                                    processed_text=row.original_lyrics,
                                    similarity=0.7 + (match_score * 0.2),
                                    match_type="lyrics_word_match"
                                ))
                
                logger.info(f"Total lyrics matches found: {len(songs)}")
                return songs
        except Exception as e:
            logger.error(f"Error in lyrics search: {e}", exc_info=True)
            return []
    
    async def get_songs_by_artist(self, artist: str, limit: int = 10) -> List[SongInfo]:
        """Get songs by artist name with proper handling of NULL values in sorting"""
        try:
            # Use SQL with text() for better control over the query
            with self.db.connection() as connection:
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
                    song = SongInfo(
                        id=row.song_id,
                        song_title=row.track_name or "",
                        artist=row.track_artist or "",
                        spotify_id=getattr(row, 'spotify_id', None),
                        popularity_score=float(row.popularity_score) if row.popularity_score is not None else 0.0,
                        similarity=0.95
                    )
                    songs.append(song)
                
                return songs
                
        except Exception as e:
            logger.error(f"Database error when searching for artist '{artist}': {e}")
            return []
    
    async def generate_llm_response(self, prompt: str, system_prompt: str) -> str:
        """Generate response using Groq"""
        if not groq_client:
            return "I'm unable to process your request at the moment."
        
        try:
            # Detect language and greeting type
            query_language = self.detect_language(prompt)
            is_thai = query_language == 'th'
            
            # Set appropriate greeting
            thai_greeting = "สวัสดีครับ"  # Default
            if is_thai and "สวัสดีค่ะ" in prompt:
                thai_greeting = "สวัสดีค่ะ"
            
            # Enhance system prompt with language and greeting instructions
            if is_thai:
                greeting_instruction = f"Always begin your response with '{thai_greeting}'"
                language_instruction = "Respond in Thai"
            else:
                greeting_instruction = "Always begin your response with 'Hello!'"
                language_instruction = "Respond in English"
                
            enhanced_system_prompt = f"""You are a music recommendation assistant specialized in songs, lyrics, and artists. 
            CRITICAL INSTRUCTIONS: 
            - {greeting_instruction}
            - {language_instruction}
            - NEVER generate or make up song lyrics or details
            - If you don't have information, simply state it's not in your database
            
            {system_prompt}"""
            
            # Call Groq API
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=1000
            )
            
            content = chat_completion.choices[0].message.content
            
            # Verify response has proper greeting
            if is_thai and not any(content.startswith(greeting) for greeting in [thai_greeting, "สวัสดี"]):
                content = f"{thai_greeting} {content}"
            elif not is_thai and not any(content.lower().startswith(greeting) for greeting in ["hello", "hi ", "hey "]):
                content = f"Hello! {content}"
                
            return content
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            
            # Fallback response
            if is_thai:
                return f"{thai_greeting} ขออภัย เกิดข้อผิดพลาดในการประมวลผลคำขอของคุณ โปรดลองอีกครั้งในภายหลัง"
            return "Hello! I'm sorry, I encountered an error processing your request. Please try again later."

    async def handle_lyrics_display(self, intent_data: Dict, mode: str, question: str, session_id: str = "default") -> AskQuestionResponse:
        """Handle lyrics display requests with improved song matching and context awareness for duplicate titles"""
        start_time = time.time()
        song_title = intent_data.get('song_title')
        artist = intent_data.get('artist')
        query_language = intent_data.get('detected_language', 'en')
        
        # Log the request details
        logger.info(f"Lyrics display request - Title: '{song_title}', Artist: '{artist}', Language: {query_language}, Mode: {mode}")
        
        # Check conversation context for previous mentions of artists
        context_artist = None
        prev_context = conversation_context.get_context(session_id)
        
        if not artist and prev_context and len(prev_context) > 0:
            # Search for most recent artist mention in context
            for ctx in reversed(prev_context):
                if ctx.get('artist'):
                    context_artist = ctx.get('artist')
                    logger.info(f"Found artist in context: '{context_artist}'")
                    break
        
        # Build search query with context awareness
        search_query = f"{song_title} by {artist}" if artist else song_title
        
        # Search for the song, passing context artist for disambiguation
        song_results = await self.search_songs(search_query, limit=5, context_artist=context_artist)  # Get extra results for disambiguation
        
        # Get the appropriate personality configuration
        personality = PERSONALITY_MODES.get(mode, PERSONALITY_MODES["buddy"])
        
        # Check if we need disambiguation (multiple songs with same title by different artists)
        needs_disambiguation = False
        if not artist and len(song_results) > 1:
            # Check if we have multiple distinct artists for the same title
            title_artists = {}
            for song in song_results:
                if song.song_title.lower() == song_title.lower():
                    title_key = song.song_title.lower()
                    if title_key not in title_artists:
                        title_artists[title_key] = set()
                    title_artists[title_key].add(song.artist.lower())
            
            # If we have multiple artists for the title, we need disambiguation
            for title, artists in title_artists.items():
                if len(artists) > 1:
                    needs_disambiguation = True
                    logger.info(f"Multiple artists ({len(artists)}) found for title '{title}', disambiguation needed")
                    break
        
        # If we need disambiguation and don't have a context artist, show options
        if needs_disambiguation and not context_artist:
            return await self._handle_title_disambiguation(
                [s for s in song_results if s.song_title.lower() == song_title.lower()], 
                song_title, 
                mode, 
                query_language
            )
        
        # Used context artist for disambiguation?
        used_context = False
        # If we have results and the first match used context for disambiguation, mark it
        if song_results and hasattr(song_results[0], 'match_type') and song_results[0].match_type == 'context_artist':
            used_context = True
            logger.info(f"Used context artist '{context_artist}' to resolve ambiguity")
        
        # Continue with normal flow
        if song_results:
            found_song = song_results[0]
            found_title = found_song.song_title
            found_artist = found_song.artist
            match_type = found_song.match_type if hasattr(found_song, 'match_type') else "unknown"
            
            # Log what we found
            logger.info(f"Found song: '{found_title}' by '{found_artist}' (match type: {match_type}, similarity: {found_song.similarity})")
            
            # Check if the found song matches what was requested
            is_exact_title_match = (found_title.lower() == song_title.lower())
            
            if not is_exact_title_match:
                logger.warning(f"Found song doesn't match requested title. Requested: '{song_title}', Found: '{found_title}'")
            
            # Get song details
            song = self.db.query(Song).filter(Song.song_id == found_song.id).first()
            
            # Check if we have lyrics
            has_lyrics = song and song.original_lyrics and len(song.original_lyrics.strip()) > 0
            logger.info(f"Has lyrics: {has_lyrics}")
            
            if has_lyrics:
                # Format lyrics
                lyrics = self.format_lyrics(song.original_lyrics)
                
                # Different handling based on match quality
                if is_exact_title_match:
                    # Exact match - show lyrics, with context clarification if needed
                    if query_language == 'th':
                        context_clarification = f"\n\nฉันใช้บริบทจากการสนทนาก่อนหน้าที่คุณพูดถึง {context_artist} และแสดงเนื้อเพลง \"{song.track_name}\" โดย {song.track_artist}" if used_context else ""
                        
                        user_prompt = f"""
                        ผู้ใช้ขอเนื้อเพลง "{song_title}"{' โดย ' + artist if artist else ''}
                        
                        พบเพลงในฐานข้อมูล:
                        ชื่อเพลง: {song.track_name}
                        ศิลปิน: {song.track_artist}{context_clarification}
                        
                        เนื้อเพลง:
                        {lyrics}
                        
                        กรุณาแสดงเนื้อเพลงในรูปแบบ {personality['response_format']}
                        สำคัญ: ทำให้ชื่อเพลงและชื่อศิลปินเป็นตัวหนา (ใช้ ** รอบชื่อ) ทุกครั้งที่กล่าวถึง
                        """
                    else:
                        context_clarification = f"\n\nI used context from your previous conversation mentioning {context_artist} to show you lyrics for \"{song.track_name}\" by {song.track_artist}" if used_context else ""
                        
                        user_prompt = f"""
                        The user requested lyrics for the song "{song_title}"{' by ' + artist if artist else ''}
                        
                        Found song in database:
                        Title: {song.track_name}
                        Artist: {song.track_artist}{context_clarification}
                        
                        Lyrics:
                        {lyrics}
                        
                        Please display these lyrics in a {personality['response_format']} style.
                        IMPORTANT: Make song titles and artist names bold (use ** around them) every time you mention them.
                        """
                else:
                    # Not an exact match - inform user and offer what we found
                    if query_language == 'th':
                        user_prompt = f"""
                        ผู้ใช้ขอเนื้อเพลง "{song_title}"{' โดย ' + artist if artist else ''}
                        
                        ไม่พบเพลงที่ตรงกับที่ขอ แต่พบเพลงที่ใกล้เคียง:
                        ชื่อเพลง: {song.track_name}
                        ศิลปิน: {song.track_artist}
                        
                        เนื้อเพลง:
                        {lyrics}
                        
                        กรุณาแจ้งผู้ใช้ว่าไม่พบเพลงที่ขอโดยตรง แต่พบเพลงที่ใกล้เคียง และแสดงเนื้อเพลงที่พบในรูปแบบ {personality['response_format']}
                        สำคัญ: ทำให้ชื่อเพลงและชื่อศิลปินเป็นตัวหนา (ใช้ ** รอบชื่อ) ทุกครั้งที่กล่าวถึง
                        """
                    else:
                        user_prompt = f"""
                        The user requested lyrics for the song "{song_title}"{' by ' + artist if artist else ''}
                        
                        The exact song was not found, but a similar one was found:
                        Title: {song.track_name}
                        Artist: {song.track_artist}
                        
                        Lyrics:
                        {lyrics}
                        
                        Please inform the user that their exact requested song wasn't found, but you found a similar one. Display the lyrics in a {personality['response_format']} style.
                        IMPORTANT: Make song titles and artist names bold (use ** around them) every time you mention them.
                        """
            else:
                # Song found but no lyrics
                if query_language == 'th':
                    user_prompt = f"""
                    ผู้ใช้ขอเนื้อเพลง "{song_title}"{' โดย ' + artist if artist else ''}
                    
                    พบเพลง "{song.track_name}" โดย {song.track_artist} ในฐานข้อมูล แต่ไม่มีเนื้อเพลง
                    
                    กรุณาแจ้งผู้ใช้ว่าพบเพลงแล้ว แต่ไม่มีเนื้อเพลงในฐานข้อมูล ในรูปแบบ {personality['response_format']}
                    สำคัญ: ทำให้ชื่อเพลงและชื่อศิลปินเป็นตัวหนา (ใช้ ** รอบชื่อ) ทุกครั้งที่กล่าวถึง
                    """
                else:
                    user_prompt = f"""
                    The user requested lyrics for the song "{song_title}"{' by ' + artist if artist else ''}
                    
                    The song "{song.track_name}" by {song.track_artist} was found in the database, but has no lyrics.
                    
                    Please inform the user that the song was found but no lyrics are available in a {personality['response_format']} style.
                    IMPORTANT: Make song titles and artist names bold (use ** around them) every time you mention them.
                    """
        else:
            # No song found at all
            logger.warning(f"No songs found matching '{search_query}'")
            
            if query_language == 'th':
                user_prompt = f"""
                ผู้ใช้ขอเนื้อเพลง "{song_title}"{' โดย ' + artist if artist else ''}
                
                ไม่พบเพลงที่ตรงกับที่ขอในฐานข้อมูล
                
                กรุณาแจ้งผู้ใช้ว่าไม่พบเพลงในฐานข้อมูล ในรูปแบบ {personality['response_format']}
                สำคัญ: ทำให้ชื่อเพลงและชื่อศิลปินเป็นตัวหนา (ใช้ ** รอบชื่อ) ทุกครั้งที่กล่าวถึง
                """
            else:
                user_prompt = f"""
                The user requested lyrics for the song "{song_title}"{' by ' + artist if artist else ''}
                
                No matching song was found in the database.
                
                Please inform the user that the song was not found in the database in a {personality['response_format']} style.
                IMPORTANT: Make song titles and artist names bold (use ** around them) every time you mention them.
                """
        
        # Generate response with LLM
        logger.info("Generating LLM response")
        response_text = await self.generate_llm_response(user_prompt, personality["system_prompt"])
        logger.info("LLM response generated successfully")
        
        return AskQuestionResponse(
            response=response_text,
            mode=mode,
            intent="lyrics_display",
            sources=[{"title": song_results[0].song_title, "artist": song_results[0].artist}] if song_results else [],
            processing_time=round(time.time() - start_time, 3)
        )

    async def _handle_title_disambiguation(self, songs: List[SongInfo], requested_title: str, mode: str, language: str) -> AskQuestionResponse:
        """Handle the case when multiple songs with the same title exist"""
        start_time = time.time()
        personality = PERSONALITY_MODES.get(mode, PERSONALITY_MODES["buddy"])
        
        # Group songs by artist to remove duplicates, case-insensitive
        artist_groups = {}
        for song in songs:
            artist_lower = song.artist.lower()
            if artist_lower not in artist_groups:
                artist_groups[artist_lower] = song
        
        # Convert back to list and sort by popularity or similarity
        unique_artist_songs = list(artist_groups.values())
        unique_artist_songs.sort(key=lambda x: getattr(x, 'similarity', 0.0), reverse=True)
        
        # Additional search to ensure we get ALL versions of the song
        try:
            # Using direct database query to find ALL songs with this title
            additional_songs = self.db.query(Song).filter(
                func.lower(Song.track_name) == func.lower(requested_title)
            ).order_by(
                case((Song.popularity_score == None, 0), else_=Song.popularity_score).desc()
            ).limit(10).all()
            
            # Convert to SongInfo and add to our list if not already there
            for song in additional_songs:
                artist_lower = song.track_artist.lower()
                if artist_lower not in artist_groups:
                    artist_groups[artist_lower] = SongInfo(
                        id=song.song_id,
                        song_title=song.track_name,
                        artist=song.track_artist,
                        processed_text=song.original_lyrics,
                        similarity=0.9,
                        match_type="db_match"
                    )
            
            # Rebuild our list with all found songs
            unique_artist_songs = list(artist_groups.values())
            unique_artist_songs.sort(key=lambda x: getattr(x, 'similarity', 0.0), reverse=True)
            
            logger.info(f"Found {len(unique_artist_songs)} unique artists for song '{requested_title}'")
            for song in unique_artist_songs:
                logger.info(f"- Artist option: {song.artist}")
        except Exception as e:
            logger.error(f"Error finding additional artist versions: {e}")
        
        # Get the song title to display from the first result
        display_title = unique_artist_songs[0].song_title if unique_artist_songs else requested_title
        
        # Format artist options with identical indentation for all entries
        artist_options_list = []
        for i, song in enumerate(unique_artist_songs[:5], 1):
            if language == 'th':
                artist_options_list.append(f"{i}. {display_title} โดย **{song.artist}**")
            else:
                artist_options_list.append(f"{i}. {display_title} by **{song.artist}**")
        
        # Join with line breaks - with no extra spacing
        artist_options = "\n".join(artist_options_list)
        
        # Create message templates with consistent formatting
        if language == 'th':
            if mode == "fun":
                message = f"สวัสดีครับ! 🎵 มีหลายเพลงที่ชื่อ \"**{display_title}**\" ในฐานข้อมูลของเรา! คุณต้องการฟังเพลงของศิลปินไหนหรอครับ? 🤔\n\n{artist_options}\n\nกรุณาระบุชื่อศิลปินที่ต้องการด้วยนะครับ! ✨"
            elif mode == "mentor":
                message = f"สวัสดีครับ ผมพบว่ามีหลายเพลงที่ชื่อ \"**{display_title}**\" ในฐานข้อมูลของเรา โดยแต่ละเพลงนั้นขับร้องโดยศิลปินที่แตกต่างกัน\n\n{artist_options}\n\nกรุณาระบุให้ชัดเจนว่าต้องการเนื้อเพลงจากศิลปินท่านใด เพื่อให้ผมสามารถให้ข้อมูลได้อย่างถูกต้องครับ"
            else:  # buddy mode
                message = f"สวัสดีครับ มีหลายเพลงที่ชื่อ \"**{display_title}**\" ในฐานข้อมูลของเรา\n\n{artist_options}\n\nกรุณาระบุชื่อศิลปินที่ต้องการด้วยนะครับ!"
        else:
            if mode == "fun":
                message = f"Hello! 🎵 There are multiple songs titled \"**{display_title}**\" in our database! Which artist's version would you like? 🤔\n\n{artist_options}\n\nPlease let me know which artist's lyrics you'd like to see! ✨"
            elif mode == "mentor":
                message = f"Hello! I've found that there are multiple songs titled \"**{display_title}**\" in our database, each performed by different artists.\n\n{artist_options}\n\nPlease specify which artist's version you're interested in, so I can provide you with the correct information."
            else:  # buddy mode
                message = f"Hello! There are multiple songs titled \"**{display_title}**\" in our database.\n\n{artist_options}\n\nWhich artist's lyrics would you like to see?"
        
        return AskQuestionResponse(
            response=message,
            mode=mode,
            intent="disambiguation",
            sources=[{"title": song.song_title, "artist": song.artist} for song in unique_artist_songs[:5]],
            processing_time=round(time.time() - start_time, 3)
        )

    def _check_title_uniqueness(self, title: str) -> bool:
        """Check if a song title is unique in the database"""
        try:
            count = self.db.query(func.count(Song.song_id)).filter(
                func.lower(Song.track_name) == func.lower(title)
            ).scalar()
            return count <= 1
        except Exception as e:
            logger.error(f"Error checking title uniqueness: {e}")
            return True  # Default to assuming it's unique

    def _has_duplicate_titles(self, songs: List[SongInfo]) -> bool:
        """Check if the list of songs contains duplicate titles with different artists"""
        if len(songs) <= 1:
            return False
            
        titles = {}
        for song in songs:
            title_lower = song.song_title.lower()
            if title_lower in titles and titles[title_lower] != song.artist.lower():
                return True
            titles[title_lower] = song.artist.lower()
        
        return False

    def format_lyrics(self, raw_lyrics: str) -> str:
        """Basic lyrics formatter"""
        if not raw_lyrics:
            return ""
        
        # Replace escaped newlines with actual newlines
        formatted = raw_lyrics.replace('\\n', '\n').replace('\\r', '\r')
        
        # Clean up excessive spacing and newlines
        formatted = re.sub(r' +', ' ', formatted)
        formatted = re.sub(r'\n{3,}', '\n\n', formatted)
        
        return formatted.strip()
    
    async def handle_lyrics_search(self, lyrics_fragment: str, mode: str, question: str) -> AskQuestionResponse:
        """Handle lyrics search queries with personality-based responses"""
        start_time = time.time()
        query_language = self.detect_language(question)
        
        # Search for lyrics
        results = await self.search_by_lyrics(lyrics_fragment, limit=3)
        
        if results:
            # Format response with the identified song
            song_info = results[0]  # Take the best match
            
            if query_language == 'th':
                greeting = "สวัสดีครับ" if "สวัสดีค่ะ" not in question else "สวัสดีค่ะ"
                
                if mode == "fun":
                    response_text = f"{greeting} 🎵 เนื้อเพลงที่คุณถามมาเป็นของเพลง \"{song_info.song_title}\" โดย {song_info.artist} นั่นเองค่ะ! 🎤"
                elif mode == "mentor":
                    response_text = f"{greeting} เนื้อเพลงที่คุณถามมาคือส่วนหนึ่งของเพลง \"{song_info.song_title}\" โดย {song_info.artist} เป็นเพลงที่มีเอกลักษณ์เฉพาะตัว"
                else:  # buddy mode
                    response_text = f"{greeting} เนื้อเพลงที่คุณถามมาเป็นของเพลง \"{song_info.song_title}\" โดย {song_info.artist} ครับ"
            else:
                if mode == "fun":
                    response_text = f"Hello! 🎵 The lyrics you asked about are from the song \"{song_info.song_title}\" by {song_info.artist}! 🎤"
                elif mode == "mentor":
                    response_text = f"Hello! The lyrics you're inquiring about are from the song \"{song_info.song_title}\" by {song_info.artist}. This is a distinctive piece in their repertoire."
                else:  # buddy mode
                    response_text = f"Hello! The lyrics you asked about are from the song \"{song_info.song_title}\" by {song_info.artist}."
        else:
            # No matching lyrics found
            if query_language == 'th':
                greeting = "สวัสดีครับ" if "สวัสดีค่ะ" not in question else "สวัสดีค่ะ"
                
                if mode == "fun":
                    response_text = f"{greeting} 🎵 ขอโทษด้วยค่ะ ฉันไม่พบเพลงที่มีเนื้อร้องนี้ในฐานข้อมูลของเรา 🔍 ลองถามเนื้อเพลงอื่นดูนะคะ!"
                else:
                    response_text = f"{greeting} ขออภัย ผมไม่พบเพลงที่มีเนื้อร้องตามที่คุณถามในฐานข้อมูลของเรา โปรดลองถามเนื้อเพลงอื่น"
            else:
                if mode == "fun":
                    response_text = "Hello! 🎵 Sorry, I couldn't find any song with those lyrics in our database 🔍 Try asking about different lyrics!"
                else:
                    response_text = "Hello! Sorry, I couldn't find any song with those lyrics in our database. Please try asking about different lyrics."
        
        return AskQuestionResponse(
            response=response_text,
            mode=mode,
            intent="lyrics_search",
            sources=[{"title": result.song_title, "artist": result.artist} for result in results],
            processing_time=round(time.time() - start_time, 3)
        )
    
    async def handle_artist_recommendations(self, artist: str, mode: str, query_language: str, limit: int = 5) -> AskQuestionResponse:
        """Create a formatted response with a list of songs by an artist"""
        start_time = time.time()
        
        # Get the personality configuration
        personality = PERSONALITY_MODES.get(mode, PERSONALITY_MODES["buddy"])
        
        # Log the request
        logger.info(f"Finding {limit} songs by artist: {artist} in {mode} mode")
        
        # Get songs from database
        songs = await self.get_songs_by_artist(artist, limit=limit)
        
        # If no songs found, use LLM to generate appropriate message
        if not songs:
            # Create system prompt with personality guidance
            system_prompt = personality["system_prompt"]
            
            # Create user prompt for no results
            if query_language == 'th':
                user_prompt = f"ผู้ใช้ขอคำแนะนำเพลงของ {artist} แต่ไม่พบในฐานข้อมูล กรุณาแจ้งว่าไม่พบเพลงของศิลปินนี้"
            else:
                user_prompt = f"The user asked for song recommendations by {artist}, but none were found in the database. Please inform them that no songs by this artist were found."
            
            # Generate response with LLM
            response_text = await self.generate_llm_response(user_prompt, system_prompt)
            
            return AskQuestionResponse(
                response=response_text,
                mode=mode,
                intent="artist_recommendations",
                sources=[],
                processing_time=round(time.time() - start_time, 3)
            )
        
        # Format songs data for LLM
        song_list = "\n".join([f"- **{song.song_title}**" for song in songs])
        
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
        response_text = await self.generate_llm_response(user_prompt, system_prompt)
        
        # Format sources for UI display
        sources = [
            {
                "title": song.song_title, 
                "artist": song.artist, 
                "similarity": 0.95,
                "match_type": "artist"
            } 
            for song in songs
        ]
        
        return AskQuestionResponse(
            response=response_text,
            mode=mode,
            intent="artist_recommendations",
            sources=sources,
            processing_time=round(time.time() - start_time, 3)
        )
        
    async def process_question(
    self, 
    question: str, 
    mode: str = "buddy",
    session_id: str = "default",
    user_id: str = "anonymous"
) -> AskQuestionResponse:
        """Main method to process music questions"""
        start_time = time.time()
        
        # Check if music-related
        if not await self.is_music_related_query(question):
            query_language = self.detect_language(question)
            personality = PERSONALITY_MODES.get(mode, PERSONALITY_MODES["buddy"])
            
            # Create system prompt for off-topic responses
            if query_language == 'th':
                thai_greeting = "สวัสดีค่ะ" if "สวัสดีค่ะ" in question else "สวัสดีครับ"
                system_prompt = f"""คุณเป็น AI ผู้ช่วยเกี่ยวกับเพลง
                คำแนะนำ:
                - เริ่มต้นด้วย "{thai_greeting}"
                - อธิบายว่าคุณเป็น AI เฉพาะด้านเพลง
                - บอกว่าไม่สามารถตอบคำถามเกี่ยวกับเรื่องอื่นได้
                - ใช้รูปแบบการตอบที่เป็น {personality["response_format"]}
                - {"ใส่ emoji ประกอบ" if mode == "fun" else "พูดด้วยท่าทีเป็นมิตร"}"""
                user_prompt = f'ผู้ใช้ถาม: "{question}"\nคำถามนี้ไม่เกี่ยวข้องกับเพลง'
            else:
                system_prompt = f"""You are an AI assistant specialized in music.
                Instructions:
                - Always start with "Hello!"
                - Explain that you are a music-specialized AI
                - Say you cannot answer questions about other topics
                - Use a {personality["response_format"]} tone
                - {"Include emojis to make it fun" if mode == "fun" else "Be friendly and helpful"}"""
                user_prompt = f'User asked: "{question}"\nThis question is not related to music.'
            
            # Generate and return off-topic response
            response_text = await self.generate_llm_response(user_prompt, system_prompt)
            
            # Save chat history if not anonymous
            if user_id != "anonymous":
                self.save_chat_history(
                    user_id=user_id,
                    session_id=session_id,
                    query=question,
                    response=response_text,
                    mode=mode,
                    intent="off_topic"
                )
            
            return AskQuestionResponse(
                response=response_text,
                mode=mode,
                intent="off_topic",
                sources=[],
                processing_time=round(time.time() - start_time, 3)
            )
        
        # Parse query intent
        context = conversation_context.get_context(session_id)
        intent_data = await self.parse_query_intent(question, context)
        
        # Update conversation context
        conversation_context.update_context(session_id, {
            'query': question,
            'intent': intent_data.get('intent'),
            'song_title': intent_data.get('song_title'),
            'artist': intent_data.get('artist')
        })
        
        # Check if this is a response to a previous disambiguation
        is_disambiguation_response = False
        previous_intent = None
        if context and len(context) > 0:
            previous_intent = context[-1].get('intent')
            if previous_intent == 'disambiguation' and intent_data.get('artist'):
                is_disambiguation_response = True
                # Get the song title from the previous context since the user is likely
                # just specifying the artist in their response
                if not intent_data.get('song_title') and context[-1].get('song_title'):
                    intent_data['song_title'] = context[-1].get('song_title')
        
        # Handle different intents
        if intent_data.get('intent') == 'lyrics_display' or is_disambiguation_response:
            result = await self.handle_lyrics_display(intent_data, mode, question, session_id)
            
        elif intent_data.get('intent') == 'lyrics_search':
            lyrics_fragment = intent_data.get('lyrics_fragment')
            if lyrics_fragment:
                result = await self.handle_lyrics_search(lyrics_fragment, mode, question)
            else:
                # No lyrics fragment provided
                query_language = intent_data.get('detected_language', 'en')
                if query_language == 'th':
                    response_text = "สวัสดีครับ กรุณาระบุเนื้อเพลงที่ต้องการค้นหา"
                else:
                    response_text = "Hello! Please provide lyrics to search for."
                
                result = AskQuestionResponse(
                    response=response_text,
                    mode=mode,
                    intent="lyrics_search",
                    sources=[],
                    processing_time=round(time.time() - start_time, 3)
                )
                
        elif intent_data.get('intent') == 'artist_recommendations' and intent_data.get('artist'):
            result = await self.handle_artist_recommendations(
                artist=intent_data.get('artist'),
                mode=mode,
                query_language=intent_data.get('detected_language', 'en'),
                limit=5
            )
            
        else:
            # Get context artist for potential disambiguation
            context_artist = None
            if context and len(context) > 0:
                for ctx in reversed(context):
                    if ctx.get('artist'):
                        context_artist = ctx.get('artist')
                        break
            
            # General query - search and respond with context awareness
            search_results = await self.search_songs(question, limit=5, context_artist=context_artist)
            
            if search_results:
                # Check if we have ambiguous results that need disambiguation
                need_disambiguation = False
                title_groups = {}
                
                # Only attempt disambiguation if the query looks like a song title request
                if intent_data.get('intent') == 'general_query' and not intent_data.get('artist'):
                    for song in search_results:
                        if hasattr(song, 'has_duplicates') and song.has_duplicates:
                            need_disambiguation = True
                            title = song.song_title.lower()
                            if title not in title_groups:
                                title_groups[title] = []
                            title_groups[title].append(song)
                
                # If we need disambiguation and haven't used context, show options
                if need_disambiguation and not context_artist:
                    # Find the best matching title
                    most_likely_title = None
                    highest_similarity = 0
                    
                    for songs in title_groups.values():
                        if songs[0].similarity > highest_similarity:
                            highest_similarity = songs[0].similarity
                            most_likely_title = songs[0].song_title
                    
                    if most_likely_title:
                        matching_songs = [s for s in search_results if s.song_title.lower() == most_likely_title.lower()]
                        return await self._handle_title_disambiguation(
                            matching_songs,
                            most_likely_title,
                            mode,
                            intent_data.get('detected_language', 'en')
                        )
                
                # Create context from results
                personality = PERSONALITY_MODES.get(mode, PERSONALITY_MODES["buddy"])
                query_language = intent_data.get('detected_language', 'en')
                
                context_items = []
                for item in search_results:
                    song = self.db.query(Song).filter(Song.song_id == item.id).first()
                    lyrics = song.original_lyrics if song and song.original_lyrics else '[NO LYRICS FOUND]'
                    context_items.append(
                        f"Song: {item.song_title} by {item.artist}\nLyrics: {lyrics}"
                    )
                
                context_text = "\n\n---\n\n".join(context_items)
                
                # Add context usage note if we used context for disambiguation
                used_context_note = ""
                if context_artist and any(s.match_type == 'context_artist' for s in search_results if hasattr(s, 'match_type')):
                    if query_language == 'th':
                        used_context_note = f"\n\nหมายเหตุสำหรับ AI: ฉันได้ใช้บริบทจากการสนทนาก่อนหน้าเกี่ยวกับศิลปิน '{context_artist}' ในการค้นหา กรุณาแจ้งผู้ใช้เกี่ยวกับการใช้บริบทนี้ในการตอบคำถาม"
                    else:
                        used_context_note = f"\n\nNote for AI: I used context from previous conversation about artist '{context_artist}' in the search. Please inform the user about this context usage in your response."
                
                if query_language == 'th':
                    prompt = f"""คำถามของผู้ใช้: "{question}"
                    
                    ข้อมูลเพลงที่พบ: 
                    {context_text}{used_context_note}
                    
                    คำแนะนำ:
                    - ตอบเป็นภาษาไทยในรูปแบบ {personality["response_format"]}
                    - เริ่มต้นด้วย "สวัสดีครับ"
                    - แสดงเฉพาะข้อมูลที่ระบุไว้ข้างต้น
                    - ห้ามแต่งเนื้อเพลงหรือข้อมูลเพิ่มเติม"""
                else:
                    prompt = f"""User query: "{question}"
                    
                    Songs found:
                    {context_text}{used_context_note}
                    
                    Instructions:
                    - Respond in English in a {personality["response_format"]} style
                    - Always start with "Hello!"
                    - ONLY display information shown above
                    - DO NOT generate any information not provided"""
                
                response_text = await self.generate_llm_response(prompt, personality["system_prompt"])
            else:
                query_language = intent_data.get('detected_language', 'en')
                if query_language == 'th':
                    response_text = "สวัสดีครับ ขออภัย ไม่พบข้อมูลเพลงที่ตรงกับคำค้นหาของคุณ"
                else:
                    response_text = "Hello! Sorry, I couldn't find any songs matching your query."
            
            result = AskQuestionResponse(
                response=response_text,
                mode=mode,
                intent=intent_data['intent'],
                sources=[{"title": song.song_title, "artist": song.artist} for song in search_results],
                processing_time=round(time.time() - start_time, 3)
            )
        
        # Save to chat history if not anonymous
        if user_id != "anonymous":
            self.save_chat_history(
                user_id=user_id,
                session_id=session_id,
                query=question,
                response=result.response,
                mode=mode,
                intent=result.intent
            )
        
        return result


    def save_chat_history(
        self,
        user_id: str,
        session_id: str,
        query: str,
        response: str,
        mode: str,
        intent: str = "general_query"
    ):
        """Save chat to database with better connection handling"""
        try:
            # Create a new session for each save operation
            chat = ChatHistory(
                user_id=user_id,
                session_id=session_id,
                query=query,
                response=response,
                mode=mode,
                intent=intent
            )
            self.db.add(chat)
            
            try:
                self.db.commit()
                return chat
            except Exception as db_error:
                self.db.rollback()
                
                # If we get a connection closed error, try one more time with a new session
                if "This Connection is closed" in str(db_error):
                    logger.warning("Connection closed, creating new session and retrying")
                    from sqlalchemy.orm import Session
                    from app.database import engine
                    
                    with Session(bind=engine) as new_session:
                        new_chat = ChatHistory(
                            user_id=user_id,
                            session_id=session_id,
                            query=query,
                            response=response,
                            mode=mode,
                            intent=intent
                        )
                        new_session.add(new_chat)
                        new_session.commit()
                        return new_chat
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"Database error in save_chat_history: {str(e)}")
            print(f"Database error in save_chat_history: {e}")
            # Don't re-raise to avoid breaking the chat flow
            return None
    
    def get_chat_history(self, session_id: str, limit: int = 50):
        """Get chat history for a session"""
        try:
            return self.db.query(ChatHistory).filter(
                ChatHistory.session_id == session_id
            ).order_by(ChatHistory.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"Error retrieving chat history: {e}")
            return []
    
    def get_user_chat_history(self, user_id: str, limit: int = 50):
        """Get chat history for a user"""
        try:
            return self.db.query(ChatHistory).filter(
                ChatHistory.user_id == user_id
            ).order_by(ChatHistory.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"Error retrieving user chat history: {e}")
            return []