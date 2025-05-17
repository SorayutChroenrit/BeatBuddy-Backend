import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str
    
    # OAuth
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    GITHUB_CLIENT_ID: str = ""
    GITHUB_CLIENT_SECRET: str = ""
    
    # JWT
    JWT_SECRET: str
    
    # URLs
    FRONTEND_URL: str
    BACKEND_URL: str
    
    # External APIs
    GROQ_API_KEY: str = ""
    
    class Config:
        env_file = ".env"

settings = Settings()