from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings
import ssl

# Get the system's default SSL certificate path
try:
    ssl_cert = ssl.get_default_verify_paths().cafile
except:
    # Fallback to common paths if automatic detection fails
    ssl_cert = "/etc/ssl/cert.pem"  # Common path on macOS

# Create engine with TiDB specific configuration
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={
        "ssl": {
            "ssl_mode": "VERIFY_IDENTITY",
            "ssl_ca": ssl_cert,
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

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class
Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)