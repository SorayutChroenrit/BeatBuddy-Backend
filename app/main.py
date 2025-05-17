import logging
import urllib.parse
import secrets  
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from sqlalchemy.orm import Session
import httpx

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

@app.get("/debug-oauth-flow")
async def debug_oauth_flow():
    """Debug the OAuth flow with detailed information"""
    google_config = OAUTH_PROVIDERS["google"]
    
    # Create a test OAuth URL
    test_state = generate_state()
    test_nonce = secrets.token_urlsafe(16)
    
    test_params = {
        "client_id": google_config["client_id"],
        "redirect_uri": google_config["redirect_uri"],
        "response_type": "code",
        "scope": google_config["scope"],
        "state": test_state,
        "nonce": test_nonce,
        "access_type": "offline",
        "prompt": "consent"
    }
    
    test_auth_url = f"{google_config['authorize_url']}?{urllib.parse.urlencode(test_params)}"
    
    # Test connectivity to Google's endpoints
    connectivity_results = {}
    urls_to_test = [
        google_config["authorize_url"],
        google_config["token_url"],
        google_config["user_info_url"]
    ]
    
    for url in urls_to_test:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                connectivity_results[url] = {
                    "status": response.status_code,
                    "reachable": True
                }
        except Exception as e:
            connectivity_results[url] = {
                "status": None,
                "reachable": False,
                "error": str(e)
            }
    
    # Create an HTML page with test buttons and debug info
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>OAuth Flow Debug</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            pre {{ background: #f4f4f4; padding: 10px; overflow-x: auto; }}
            .section {{ margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
            button {{ padding: 10px 15px; background: #4285f4; color: white; border: none; cursor: pointer; }}
            button:hover {{ background: #3367d6; }}
        </style>
    </head>
    <body>
        <h1>OAuth Flow Debug</h1>
        
        <div class="section">
            <h2>1. Test OAuth URL</h2>
            <p>This URL should redirect you to Google's authentication page:</p>
            <pre>{test_auth_url}</pre>
            <p><a href="{test_auth_url}" target="_blank"><button>Test Google OAuth</button></a></p>
        </div>
        
        <div class="section">
            <h2>2. Connectivity Tests</h2>
            <p>Testing connectivity to Google's OAuth endpoints:</p>
            <pre>{connectivity_results}</pre>
        </div>
        
        <div class="section">
            <h2>3. OAuth Configuration</h2>
            <p>Your current OAuth configuration:</p>
            <pre>
Client ID: {google_config["client_id"][:10]}...
Redirect URI: {google_config["redirect_uri"]}
Scope: {google_config["scope"]}
Authorize URL: {google_config["authorize_url"]}
Token URL: {google_config["token_url"]}
User Info URL: {google_config["user_info_url"]}
            </pre>
        </div>
        
        <div class="section">
            <h2>4. Testing Direct Callback</h2>
            <p>Add test query parameters to your callback URL to test if it's working:</p>
            <a href="{google_config["redirect_uri"]}?code=test_code&state={test_state}" target="_blank">
                <button>Test Callback URL Directly</button>
            </a>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)