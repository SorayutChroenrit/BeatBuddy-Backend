import secrets
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, Depends
from sqlalchemy.orm import Session
import httpx

from app.config import settings
from app.database import get_db
from app.models import User, Account

# OAuth providers
OAUTH_PROVIDERS = {
    "google": {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",  
        "token_url": "https://oauth2.googleapis.com/token",
        "user_info_url": "https://www.googleapis.com/oauth2/v2/userinfo",
        "scope": "email profile",  
        "redirect_uri": "https://beatbuddy-backend-production.up.railway.app/api/auth/callback/google"
    },
    "github": {
        "client_id": settings.GITHUB_CLIENT_ID,
        "client_secret": settings.GITHUB_CLIENT_SECRET,
        "authorize_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "user_info_url": "https://api.github.com/user",
        "scope": "user:email",
        "redirect_uri": "https://beatbuddy-backend-production.up.railway.app/api/auth/callback/github"
    }
}

# JWT functions
def create_jwt_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(days=7),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")

def verify_jwt_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        return payload.get("user_id")
    except:
        return None

# Auth dependency
async def get_current_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    session_token = request.cookies.get("session-token")
    if not session_token:
        return None
    
    user_id = verify_jwt_token(session_token)
    if not user_id:
        return None
    
    return db.query(User).filter(User.id == user_id).first()

# OAuth functions
def generate_state() -> str:
    return secrets.token_urlsafe(32)

async def exchange_code_for_token(provider: str, code: str, redirect_uri: str) -> Dict[str, Any]:
    """Exchange authorization code for access token with better error handling"""
    provider_config = OAUTH_PROVIDERS[provider]
    
    data = {
        "grant_type": "authorization_code",
        "client_id": provider_config["client_id"],
        "client_secret": provider_config["client_secret"],
        "code": code,
        "redirect_uri": redirect_uri
    }
    
    headers = {"Accept": "application/json"}
    
    try:
        print(f"Exchanging code for token with provider: {provider}")
        print(f"Using redirect URI: {redirect_uri}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                provider_config["token_url"],
                data=data,
                headers=headers
            )
            
            status_code = response.status_code
            content_type = response.headers.get("content-type", "")
            response_text = response.text
            
            print(f"Token exchange response status: {status_code}")
            print(f"Token exchange content type: {content_type}")
            
            if status_code != 200:
                print(f"Token exchange error: {response_text}")
                return {}
            
            # Handle different response formats
            if "application/json" in content_type:
                return response.json()
            elif "application/x-www-form-urlencoded" in content_type:
                # Parse form-urlencoded response
                result = {}
                for item in response_text.split("&"):
                    if "=" in item:
                        key, value = item.split("=", 1)
                        result[key] = value
                return result
            else:
                print(f"Unexpected content type: {content_type}")
                try:
                    return response.json()
                except:
                    print(f"Could not parse response as JSON: {response_text}")
                    return {}
    except Exception as e:
        print(f"Error exchanging code for token: {str(e)}")
        return {}
    
async def get_user_info(provider: str, access_token: str) -> Dict[str, Any]:
    provider_config = OAUTH_PROVIDERS[provider]
    headers = {"Authorization": f"Bearer {access_token}"}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(provider_config["user_info_url"], headers=headers)
        user_data = response.json()
        
        # Handle GitHub email
        if provider == "github" and not user_data.get("email"):
            email_response = await client.get("https://api.github.com/user/emails", headers=headers)
            if email_response.status_code == 200:
                emails = email_response.json()
                primary_email = next((e for e in emails if e["primary"]), None)
                if primary_email:
                    user_data["email"] = primary_email["email"]
        
        return user_data

def create_or_update_user(db: Session, user_data: Dict[str, Any], provider: str, provider_id: str) -> User:
    email = user_data.get("email")
    
    # Find existing user
    user = None
    if email:
        user = db.query(User).filter(User.email == email).first()
    
    # Create new user
    if not user:
        user = User(
            email=email,
            name=user_data.get("name"),
            image=user_data.get("picture") or user_data.get("avatar_url")
        )
        db.add(user)
        db.flush()
    
    # Create/update account
    account = db.query(Account).filter(
        Account.provider == provider,
        Account.providerAccountId == provider_id
    ).first()
    
    if not account:
        account = Account(
            userId=user.id,
            type="oauth",
            provider=provider,
            providerAccountId=provider_id,
            access_token=user_data.get("access_token"),
            refresh_token=user_data.get("refresh_token")
        )
        db.add(account)
    
    db.commit()
    return user