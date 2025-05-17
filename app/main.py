import logging
import urllib.parse
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy.orm import Session

# Local imports
from app.config import settings
from app.database import get_db, create_tables
from app.models import User
from app.schemas import (
    AskQuestionRequest, 
    AskQuestionResponse, 
    ChatMessage, 
    SessionData,
    User as UserSchema
)
from app.auth import (
    get_current_user,
    generate_state,
    OAUTH_PROVIDERS,
    exchange_code_for_token,
    get_user_info,
    create_or_update_user,
    create_jwt_token
)
from app.music import MusicService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Music RAG API", version="1.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Music endpoints
@app.post("/ask", response_model=AskQuestionResponse)
async def ask_question(
    request_data: AskQuestionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Main music Q&A endpoint"""
    user_id = current_user.id if current_user else "anonymous"
    
    music_service = MusicService(db)
    result = await music_service.process_question(
        question=request_data.question,
        mode=request_data.mode,
        session_id=request_data.session_id,
        user_id=user_id
    )
    
    return result

@app.post("/save-chat")
async def save_chat(
    message: ChatMessage,
    db: Session = Depends(get_db)
):
    """Save chat message"""
    try:
        print(f"Saving chat: {message.dict()}")
        music_service = MusicService(db)
        music_service.save_chat_history(
            user_id=message.user_id,
            session_id=message.session_id,
            query=message.query,
            response=message.response,
            mode=message.mode
        )
        return {"status": "saved"}
    except Exception as e:
        print(f"Error saving chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving chat: {str(e)}")

@app.get("/chat-history/{session_id}")
async def get_direct_session_history(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Direct chat history for session - for backward compatibility"""
    music_service = MusicService(db)
    history = music_service.get_chat_history(session_id)
    return {"messages": [msg.__dict__ for msg in history]}


@app.get("/chat-history/user/{user_id}")  
async def get_user_chat_history_endpoint(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get chat history for user"""
    music_service = MusicService(db)
    history = music_service.get_user_chat_history(user_id)
    return {"messages": [msg.to_dict() for msg in history]}


@app.get("/chat-history/session/{session_id}")
async def get_session_history(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get chat history for session"""
    music_service = MusicService(db)
    history = music_service.get_chat_history(session_id)
    return {"messages": [msg.to_dict() for msg in history]}

# Auth endpoints
@app.get("/api/auth/session", response_model=SessionData)
async def get_session(current_user: User = Depends(get_current_user)):
    """Get current session"""
    if not current_user:
        return SessionData(user=None)
    
    return SessionData(user=UserSchema(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        image=current_user.image
    ))


@app.get("/api/auth/signin/{provider}")
async def signin(provider: str, request: Request):
    """Start OAuth flow"""
    if provider not in OAUTH_PROVIDERS:
        raise HTTPException(status_code=400, detail="Invalid provider")
    
    config = OAUTH_PROVIDERS[provider]
    state = generate_state()
    
    host = request.headers.get("host", settings.BACKEND_URL.replace("https://", "").replace("http://", ""))
    scheme = request.headers.get("x-forwarded-proto", "https" if settings.BACKEND_URL.startswith("https") else "http")
    redirect_uri = f"{scheme}://{host}/api/auth/callback/{provider}"
    
    # Debug information
    print(f"DEBUG: OAuth signin request for {provider}")
    print(f"DEBUG: Host header: {host}")
    print(f"DEBUG: Scheme: {scheme}")
    print(f"DEBUG: Constructed redirect_uri: {redirect_uri}")
    print(f"DEBUG: Client ID being used: {config['client_id'][:5]}...")
    
    # Use different endpoints for different providers
    base_url = config["authorize_url"]
    if provider == "google":
        # Use the working Google OAuth URL structure
        base_url = "https://accounts.google.com/o/oauth2/v2/auth/oauthchooseaccount"
        
        params = {
            "client_id": config["client_id"],
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "email",  # Simplified scope
            "access_type": "offline",
            "service": "lso",
            "o2v": "2",
            "flowName": "GeneralOAuthFlow",
            "state": state
        }
    else:
        # For other providers, use the original approach
        params = {
            "client_id": config["client_id"],
            "redirect_uri": redirect_uri,
            "scope": config["scope"],
            "response_type": "code",
            "state": state
        }
    
    auth_url = f"{base_url}?{urllib.parse.urlencode(params)}"
    print(f"DEBUG: Full auth URL: {auth_url}")
    
    # Create response with redirect
    response = RedirectResponse(url=auth_url)
    response.set_cookie(f"oauth_state_{provider}", state, httponly=True, max_age=600)
    
    return response
@app.get("/api/auth/callback/{provider}")
async def oauth_callback(
    provider: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Handle OAuth callback"""
    print(f"\n======= OAUTH CALLBACK =======")
    print(f"Provider: {provider}")
    print(f"URL: {request.url}")
    print(f"Query params: {dict(request.query_params)}")
    
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    stored_state = request.cookies.get(f"oauth_state_{provider}")
    
    print(f"Code present: {bool(code)}")
    print(f"State present: {bool(state)}")
    print(f"Stored state present: {bool(stored_state)}")
    print(f"States match: {state == stored_state}")
    
    if not code or not state or state != stored_state:
        error_detail = []
        if not code:
            error_detail.append("Missing authorization code")
        if not state:
            error_detail.append("Missing state parameter")
        if not stored_state:
            error_detail.append("Missing stored state cookie")
        elif state != stored_state:
            error_detail.append(f"State mismatch: received {state} but expected {stored_state}")
        
        print(f"OAuth validation failed: {', '.join(error_detail)}")
        raise HTTPException(status_code=400, detail="Invalid OAuth response: " + ", ".join(error_detail))
    
    try:
        # Continue with your original code
        redirect_uri = f"{request.base_url}api/auth/callback/{provider}"
        print(f"Using redirect URI: {redirect_uri}")
        
        token_data = await exchange_code_for_token(provider, code, redirect_uri)
        access_token = token_data.get("access_token")
        
        if not access_token:
            print("Failed to get access token")
            raise HTTPException(status_code=400, detail="Failed to get access token")
        
        print("Successfully obtained access token")
        
        # Get user info
        user_info = await get_user_info(provider, access_token)
        provider_id = str(user_info.get("id"))
        
        # Create/update user
        user = create_or_update_user(db, user_info, provider, provider_id)
        
        # Create session
        session_token = create_jwt_token(user.id)
        
        # Redirect to frontend
        response = RedirectResponse(url=f"{settings.FRONTEND_URL}/#/auth/callback")
        response.set_cookie("session-token", session_token, httponly=True, max_age=7*24*60*60)
        response.delete_cookie(f"oauth_state_{provider}")
        
        print(f"Authentication successful, redirecting to: {settings.FRONTEND_URL}/#/auth/callback")
        return response
    except Exception as e:
        print(f"Exception in OAuth callback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Authentication error: {str(e)}")
    
@app.get("/api/debug/config")
async def debug_config():
    """Debug endpoint to check configuration"""
    return {
        "google_client_id_configured": bool(settings.GOOGLE_CLIENT_ID),
        "google_client_id": settings.GOOGLE_CLIENT_ID[:5] + "..." if settings.GOOGLE_CLIENT_ID else None,
        "frontend_url": settings.FRONTEND_URL,
    }

@app.post("/api/auth/signout")
async def signout():
    """Sign out user"""
    response = JSONResponse({"message": "Signed out"})
    response.delete_cookie("session-token")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)