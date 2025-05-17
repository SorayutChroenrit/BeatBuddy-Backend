import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "mysql+pymysql://27tLCQSVFsGqhJ9.root:HVSvJQWvox3NSgeS@gateway01.ap-southeast-1.prod.aws.tidbcloud.com:4000/music_recommendation"
    
    # OAuth
    GOOGLE_CLIENT_ID: str 
    GOOGLE_CLIENT_SECRET: str 
    GITHUB_CLIENT_ID: str = ""
    GITHUB_CLIENT_SECRET: str = ""
    
    # JWT
    JWT_SECRET: str = "BEATBUDDY_SECRET"
    
    # CORS
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")
    
    # External APIs
    GROQ_API_KEY: str = ""
    
    class Config:
        env_file = ".env"

settings = Settings()