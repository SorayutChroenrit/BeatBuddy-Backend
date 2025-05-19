from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError, DBAPIError, DisconnectionError
from app.config import settings
import ssl
import logging
import time
import contextlib

logger = logging.getLogger(__name__)

# Get the system's default SSL certificate path
try:
    ssl_cert = ssl.get_default_verify_paths().cafile
except:
    # Fallback to common paths if automatic detection fails
    ssl_cert = "/etc/ssl/cert.pem"  # Common path on macOS

# connection parameters for TiDB
connect_args = {
    "ssl": {
        "ssl_mode": "VERIFY_IDENTITY",
        "ssl_ca": ssl_cert,
    },
    "connect_timeout": 120,  
    "read_timeout": 120,     
    "write_timeout": 120,    
    "autocommit": True      
}

# Create engine with more robust TiDB connection settings
engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_recycle=900,        
    pool_pre_ping=True,
    echo_pool=True,          
    pool_size=10,            
    max_overflow=20,         
    pool_timeout=60,        
    execution_options={"isolation_level": "READ COMMITTED"}  
)

# Create session with robust connection handling
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class
Base = declarative_base()

# Enhanced dependency with retry capability
def get_db(max_retries=3):
    """Get a database session with retry capability"""
    retry_count = 0
    last_error = None

    while retry_count < max_retries:
        db = SessionLocal()
        try:
            # Test connection before returning
            db.execute(text("SELECT 1"))
            logger.debug("Database connection established successfully")
            yield db
            break
        except (OperationalError, DBAPIError, DisconnectionError) as e:
            retry_count += 1
            last_error = e
            
            # Close current connection
            try:
                db.close()
            except:
                pass
            
            if retry_count < max_retries:
                # Exponential backoff
                backoff_time = 0.5 * (2 ** (retry_count - 1))
                logger.warning(f"Database connection error, retry {retry_count}/{max_retries} after {backoff_time}s: {str(e)}")
                time.sleep(backoff_time)
            else:
                logger.error(f"Failed to connect to database after {max_retries} attempts: {str(e)}")
                raise
        except Exception as e:
            # For other errors, just log and reraise
            logger.error(f"Unexpected database error: {str(e)}")
            try:
                db.close()
            except:
                pass
            raise
        finally:
            db.close()

# Context manager for database sessions
@contextlib.contextmanager
def get_db_context(max_retries=3):
    """Context manager for database connections with retry logic"""
    retry_count = 0
    last_error = None
    db = None

    while retry_count < max_retries:
        try:
            db = SessionLocal()
            # Test connection
            db.execute("SELECT 1")
            yield db
            db.commit()
            break
        except (OperationalError, DBAPIError, DisconnectionError) as e:
            if db:
                db.rollback()
            
            retry_count += 1
            last_error = e
            
            if retry_count < max_retries:
                backoff_time = 0.5 * (2 ** (retry_count - 1))
                logger.warning(f"Database connection error in context manager, retry {retry_count}/{max_retries} after {backoff_time}s: {str(e)}")
                time.sleep(backoff_time)
            else:
                logger.error(f"Failed to connect to database after {max_retries} attempts: {str(e)}")
                raise
        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"Unexpected database error in context manager: {str(e)}")
            raise
        finally:
            if db:
                db.close()

# Function to create all tables
def create_tables():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise