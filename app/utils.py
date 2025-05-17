import re
import time
from typing import Optional

def detect_language(text: str) -> str:
    """Simple language detection"""
    # Count Thai characters
    thai_chars = len(re.findall(r'[\u0E00-\u0E7F]', text))
    if thai_chars / len(text) > 0.15:
        return 'th'
    return 'en'

def format_time(seconds: float) -> str:
    """Format processing time"""
    return f"{seconds:.3f}s"

def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Basic input sanitization"""
    if not text:
        return ""
    
    # Remove null bytes and limit length
    sanitized = text.replace('\x00', '')[:max_length]
    return sanitized.strip()

def extract_song_count(query: str) -> Optional[int]:
    """Extract number of songs requested"""
    # Look for patterns like "5 songs" or "ten songs"
    number_patterns = [
        r'\b(\d+)\s+songs?\b',
        r'\b(\d+)\s+เพลง\b'
    ]
    
    for pattern in number_patterns:
        match = re.search(pattern, query.lower())
        if match:
            count = int(match.group(1))
            return min(10, max(1, count))
    
    return None