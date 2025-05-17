import logging
import urllib.parse
import secrets  
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
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

# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


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
        logger.info(f"Saving chat message")
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
        logger.error(f"Error saving chat: {e}")
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
    """Start OAuth flow with improved security measures"""
    if provider not in OAUTH_PROVIDERS:
        raise HTTPException(status_code=400, detail="Invalid provider")
    
    config = OAUTH_PROVIDERS[provider]
    state = generate_state()
    
    # Generate a nonce for extra security
    nonce = secrets.token_urlsafe(16)
    
    # Use the pre-configured redirect URI
    redirect_uri = config["redirect_uri"]

    # Create OAuth parameters with enhanced security
    params = {
        "client_id": config["client_id"],
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": config["scope"],
        "state": state,
        "nonce": nonce 
    }
    
    # Add provider-specific parameters
    if provider == "google":
        params["access_type"] = "offline"
        # Change from select_account to consent to reduce phishing fingerprint
        params["prompt"] = "consent"  
    
    # Construct the full authorization URL
    auth_url = f"{config['authorize_url']}?{urllib.parse.urlencode(params)}"
    logger.info(f"Starting OAuth flow for {provider}")
    logger.debug(f"Full auth URL: {auth_url}")
    
    # Create response with redirect and secure cookies
    response = RedirectResponse(url=auth_url)
    response.set_cookie(
        key=f"oauth_state_{provider}",
        value=state,
        httponly=True,
        secure=True,  
        samesite="lax",  
        max_age=300      
    )
    
    # Add extra security headers directly to this response
    response.headers["Cache-Control"] = "no-store, max-age=0"
    
    return response

@app.get("/api/auth/callback/{provider}")
async def oauth_callback(
    provider: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Handle OAuth callback with improved security"""
    logger.info(f"OAuth callback received for {provider}")
    
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    stored_state = request.cookies.get(f"oauth_state_{provider}")
    
    logger.debug(f"Code present: {bool(code)}")
    logger.debug(f"State present: {bool(state)}")
    logger.debug(f"Stored state present: {bool(stored_state)}")
    logger.debug(f"States match: {state == stored_state}")
    
    # Thorough state validation
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
        
        logger.warning(f"OAuth validation failed: {', '.join(error_detail)}")
        raise HTTPException(status_code=400, detail="Invalid OAuth response: " + ", ".join(error_detail))
    
    try:
        # Use the pre-configured redirect URI
        config = OAUTH_PROVIDERS[provider]
        redirect_uri = config["redirect_uri"]
        
        logger.debug(f"Using fixed redirect URI: {redirect_uri}")
        
        # Exchange code for token
        token_data = await exchange_code_for_token(provider, code, redirect_uri)
        access_token = token_data.get("access_token")
        
        if not access_token:
            logger.error(f"Failed to get access token: {token_data}")
            raise HTTPException(status_code=400, detail="Failed to get access token")
        
        logger.info("Successfully obtained access token")
        
        # Get user info
        user_info = await get_user_info(provider, access_token)
        
        # Get provider ID based on the provider
        if provider == "google":
            provider_id = user_info.get("id") or user_info.get("sub")
        else:
            provider_id = str(user_info.get("id"))
        
        if not provider_id:
            logger.error(f"Missing provider ID in user info: {user_info}")
            raise HTTPException(status_code=400, detail="Failed to get user ID from provider")
        
        # Create/update user
        user = create_or_update_user(db, user_info, provider, provider_id)
        
        # Create session token
        session_token = create_jwt_token(user.id)
        
        # Important: Make sure the redirect URL format is correct
        frontend_callback_url = f"{settings.FRONTEND_URL}/BeatBuddy/#/auth/callback"
        
        # Redirect to frontend with secure cookies
        response = RedirectResponse(url=frontend_callback_url)
        
        # Set session cookie with improved security settings
        response.set_cookie(
            key="session-token", 
            value=session_token, 
            httponly=True,
            secure=True,  
            samesite="lax",  
            max_age=7*24*60*60  # 7 days
        )
        
        # Clean up the state cookie
        response.delete_cookie(f"oauth_state_{provider}")
        
        # Add extra security headers
        response.headers["Cache-Control"] = "no-store, max-age=0"
        
        logger.info(f"Authentication successful, redirecting to: {frontend_callback_url}")
        return response
        
    except Exception as e:
        logger.error(f"Exception in OAuth callback: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Authentication error. Please try again later.")

@app.post("/api/auth/signout")
async def signout():
    """Sign out user"""
    response = JSONResponse({"message": "Signed out"})
    response.delete_cookie("session-token")
    return response

@app.get("/google{rest_of_path:path}")
async def google_verification(rest_of_path: str):
    file_content = """google-site-verification: google50108aa7baf4f0ce.html"""
    return HTMLResponse(content=file_content)

@app.get("/test-google-auth")
async def test_google_auth():
    """Test Google OAuth with manual parameters"""
    client_id = settings.GOOGLE_CLIENT_ID
    redirect_uri = "https://beatbuddy-backend-zvso.onrender.com/api/auth/callback/google"
    
    # Encode the redirect URI properly
    encoded_redirect = urllib.parse.quote(redirect_uri, safe='')
    
    # Build the URL with correct parameters
    auth_url = (
        f"https://accounts.google.com/oauth2/v2/auth?"
        f"client_id={client_id}&"
        f"redirect_uri={encoded_redirect}&"
        f"response_type=code&"
        f"scope=email&"
        f"state=test12345&"
        f"access_type=offline&"
        f"prompt=consent"
    )
    
    # Return both the URL and a page with a link
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Test Google Auth</title></head>
    <body>
        <h1>Test Google Auth</h1>
        <p>URL: {auth_url}</p>
        <a href="{auth_url}" target="_blank">Click here to test Google Auth</a>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/test-google-config")
async def test_google_config():
    """Show Google OAuth configuration"""
    return {
        "client_id": settings.GOOGLE_CLIENT_ID[:10] + "..." if settings.GOOGLE_CLIENT_ID else None,
        "client_id_length": len(settings.GOOGLE_CLIENT_ID) if settings.GOOGLE_CLIENT_ID else 0,
        "redirect_uri": OAUTH_PROVIDERS["google"]["redirect_uri"] if "redirect_uri" in OAUTH_PROVIDERS["google"] else None,
        "redirect_url": OAUTH_PROVIDERS["google"]["redirect_url"] if "redirect_url" in OAUTH_PROVIDERS["google"] else None,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)