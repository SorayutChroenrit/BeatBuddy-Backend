import json
import pandas as pd
import certifi
from sqlalchemy import create_engine, text, inspect
import pymysql
import logging
import numpy as np
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Explicitly tell SQLAlchemy to use PyMySQL
pymysql.install_as_MySQLdb()

try:
    # 1. Load the processed JSON file
    logger.info("Loading JSON data...")
    with open('data/db-huggingface/huggingface_processed_songs.json', 'r') as f:
        song_records = json.load(f)
    logger.info(f"Loaded {len(song_records)} records from JSON")
    
    # Check for long links and truncate if needed
    max_link_length = 1000  # Slightly less than field size to be safe
    long_links_count = 0
    
    for record in song_records:
        if 'link' in record and record['link'] and len(record['link']) > max_link_length:
            record['link'] = record['link'][:max_link_length]
            long_links_count += 1
    
    if long_links_count > 0:
        logger.info(f"Truncated {long_links_count} long links to fit in the database field")

    # 2. TiDB Connection Configuration
    connection_string = "mysql+pymysql://27tLCQSVFsGqhJ9.root:HVSvJQWvox3NSgeS@gateway01.ap-southeast-1.prod.aws.tidbcloud.com:4000/music_recommendation"

    # Create engine with SSL configuration
    logger.info("Connecting to TiDB with SQLAlchemy + PyMySQL...")
    engine = create_engine(
        connection_string,
        connect_args={
            "ssl": {"ca": certifi.where()},
            "ssl_verify_identity": True,
            "connect_timeout": 60,
            "read_timeout": 60,
            "write_timeout": 60
        },
        isolation_level="READ COMMITTED",
        pool_recycle=1800,
        pool_pre_ping=True
    )

    # 3. Prepare DataFrames for each table
    logger.info("Preparing DataFrames...")
    songs_df = pd.DataFrame([{
        'song_id': record['song_id'],
        'track_name': record['track_name'],
        'track_artist': record['track_artist'],
        'track_album_name': record['track_album_name'],
        'original_lyrics': record['original_lyrics'], 
        'cleaned_lyrics': record['lyrics'],
        'playlist_genre': record['playlist_genre'],
        'playlist_subgenre': record['playlist_subgenre'],
        'language': record['language'],
        'sentiment': record['sentiment'],
        'popularity_score': record['popularity_score'],
        'link': record['link'],
        'spotify_id': record['spotify_id'],
        'siamzone_id': record['siamzone_id']
    } for record in song_records])

    # Check embedding dimensions
    embedding_dimensions = []
    for record in song_records:
        if 'lyrics_embedding' in record and record['lyrics_embedding']:
            embedding_dimensions.append(len(record['lyrics_embedding']))
            break
    
    if not embedding_dimensions:
        logger.error("No embedding vectors found in data. Cannot determine dimension.")
        raise ValueError("Missing embedding data")
    
    vector_dimension = embedding_dimensions[0]
    logger.info(f"Detected embedding dimension: {vector_dimension}")
    
    # 4. Check if tables exist and handle properly with transactions
    with engine.begin() as connection:
        # Using inspector to check if tables exist
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        # If tables exist, drop them entirely to ensure schema changes take effect
        logger.info("Checking if tables need to be dropped and recreated...")
        connection.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        
        if 'song_embeddings_vector' in tables:
            connection.execute(text("DROP TABLE song_embeddings_vector"))
            logger.info("Dropped song_embeddings_vector table")
        
        if 'song_features' in tables:
            connection.execute(text("DROP TABLE song_features"))
            logger.info("Dropped song_features table")
        
        if 'songs' in tables:
            connection.execute(text("DROP TABLE songs"))
            logger.info("Dropped songs table")
        
        connection.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
        
        # Create tables regardless (IF NOT EXISTS will handle if they're already there)
        logger.info("Creating tables if they don't exist...")
        # Create tables in proper order with VECTOR type for embeddings and individual audio features
        sql_create_table = f"""
        CREATE TABLE IF NOT EXISTS songs (
            song_id VARCHAR(36) PRIMARY KEY,
            track_name VARCHAR(255),
            track_artist VARCHAR(255),
            track_album_name VARCHAR(255),
            original_lyrics TEXT,  
            cleaned_lyrics TEXT,   
            playlist_genre VARCHAR(100),
            playlist_subgenre VARCHAR(100),
            language VARCHAR(50),
            sentiment VARCHAR(50),
            popularity_score FLOAT,
            link VARCHAR(1024),
            spotify_id VARCHAR(64),
            siamzone_id VARCHAR(64)
        );

        CREATE TABLE IF NOT EXISTS song_embeddings_vector (
            song_id VARCHAR(36) PRIMARY KEY,
            lyrics_embedding VECTOR({vector_dimension}) NOT NULL,
            metadata_embedding VECTOR({vector_dimension}) NOT NULL,
            FOREIGN KEY (song_id) REFERENCES songs(song_id)
        );

        CREATE TABLE IF NOT EXISTS song_features (
            song_id VARCHAR(36) PRIMARY KEY,
            danceability FLOAT,
            energy FLOAT,
            `key` FLOAT,
            loudness FLOAT,
            mode FLOAT,
            speechiness FLOAT,
            acousticness FLOAT,
            instrumentalness FLOAT,
            liveness FLOAT,
            valence FLOAT,
            tempo FLOAT,
            FOREIGN KEY (song_id) REFERENCES songs(song_id)
        );

        -- Create regular indexes
        CREATE INDEX IF NOT EXISTS idx_spotify_id ON songs(spotify_id);
        CREATE INDEX IF NOT EXISTS idx_siamzone_id ON songs(siamzone_id);
        CREATE INDEX IF NOT EXISTS idx_language ON songs(language);
        CREATE INDEX IF NOT EXISTS idx_genre ON songs(playlist_genre);
        """
        
        for statement in sql_create_table.split(';'):
            if statement.strip():
                try:
                    connection.execute(text(statement))
                except Exception as e:
                    logger.warning(f"Statement execution warning (may be normal): {str(e)}")
                    continue
        
        # Skip vector index creation - TiDB handles this automatically
        logger.info("Skipping vector index creation - will be created automatically or later manually")

    # 5. Load data into TiDB tables with error handling and smaller chunks
    logger.info("Loading data into songs table...")
    chunksize = 100  # Even smaller chunks for better transaction handling
    
    # Use smaller chunks and error handling for loading songs
    try:
        songs_df.to_sql('songs', engine, if_exists='append', index=False, chunksize=chunksize)
        logger.info(f"Loaded {len(songs_df)} records into songs table")
    except Exception as e:
        logger.error(f"Error loading songs table: {str(e)}")
        raise
    
    # Reset connection for the next table load
    engine.dispose()
    
    # Get a raw connection to use with PyMySQL
    conn = pymysql.connect(
        host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
        port=4000,
        user="27tLCQSVFsGqhJ9.root",
        password="HVSvJQWvox3NSgeS",
        database="music_recommendation",
        ssl={"ca": certifi.where()},
        ssl_verify_identity=True,
        connect_timeout=180,
        read_timeout=180,
        write_timeout=180,
        autocommit=False,
        max_allowed_packet=50*1024*1024  # 50MB
    )
    
    # Function to format vectors in a TiDB compatible format
    def format_vector_values(vector):
        if vector is None or len(vector) == 0:
            return ""
        return ",".join([f"{v:.6f}" for v in vector])
    
    try:
        cursor = conn.cursor()
        
        # Vector embedding loading - using PyMySQL directly
        logger.info("Loading vector embeddings using PyMySQL directly...")
        
        # Split into smaller chunks for processing
        batch_size = 10
        total_loaded = 0
        total_batches = (len(song_records) + batch_size - 1) // batch_size
        
        for i in range(0, len(song_records), batch_size):
            batch = song_records[i:i+batch_size]
            logger.info(f"Processing vector embeddings batch {i//batch_size + 1}/{total_batches} (records {i+1}-{min(i+batch_size, len(song_records))})")
            
            for record in batch:
                try:
                    song_id = record.get('song_id')
                    if not song_id:
                        logger.warning(f"Skipping record - missing song_id")
                        continue
                    
                    # Get vector data
                    lyrics_embedding = record.get('lyrics_embedding')
                    metadata_embedding = record.get('metadata_embedding')
                    
                    if not lyrics_embedding or not metadata_embedding:
                        logger.warning(f"Skipping {song_id} - missing embedding data")
                        continue
                    
                    # Format vectors as comma-separated strings
                    lyrics_values = format_vector_values(lyrics_embedding)
                    metadata_values = format_vector_values(metadata_embedding)
                    
                    # Create SQL with correct TiDB VECTOR syntax
                    # According to TiDB docs, vectors should be inserted with the '[value1, value2, ...]' format
                    sql = f"""
                    INSERT INTO song_embeddings_vector 
                    (song_id, lyrics_embedding, metadata_embedding)
                    VALUES (
                        %s, 
                        '[{lyrics_values}]', 
                        '[{metadata_values}]'
                    )
                    """
                    
                    # Execute with parameter binding only for song_id (safer)
                    cursor.execute(sql, (song_id,))
                    
                    total_loaded += 1
                    
                    # Commit every 10 records to avoid transaction timeout
                    if total_loaded % 10 == 0:
                        conn.commit()
                        
                except Exception as e:
                    logger.error(f"Error inserting vector for {record.get('song_id', 'unknown')}: {str(e)}")
                    continue
            
            # Commit after each batch
            conn.commit()
            
            # Brief pause to prevent overwhelming the database
            time.sleep(0.1)
        
        logger.info(f"Loaded {total_loaded} records into song_embeddings_vector table")
        
        # Load individual audio features
        logger.info("Loading individual audio features...")
        
        total_features_loaded = 0
        
        for i in range(0, len(song_records), batch_size):
            batch = song_records[i:i+batch_size]
            logger.info(f"Processing audio features batch {i//batch_size + 1}/{total_batches} (records {i+1}-{min(i+batch_size, len(song_records))})")
            
            for record in batch:
                try:
                    song_id = record.get('song_id')
                    if not song_id:
                        logger.warning(f"Skipping record - missing song_id")
                        continue
                    
                    # Get individual audio features
                    danceability = record.get('danceability', 0.0)
                    energy = record.get('energy', 0.0)
                    key = record.get('key', 0.0)
                    loudness = record.get('loudness', 0.0)
                    mode = record.get('mode', 0.0)
                    speechiness = record.get('speechiness', 0.0)
                    acousticness = record.get('acousticness', 0.0)
                    instrumentalness = record.get('instrumentalness', 0.0)
                    liveness = record.get('liveness', 0.0)
                    valence = record.get('valence', 0.0)
                    tempo = record.get('tempo', 0.0)
                    
                    # Create SQL to insert individual features
                    sql = """
                    INSERT INTO song_features 
                    (song_id, danceability, energy, `key`, loudness, mode, 
                     speechiness, acousticness, instrumentalness, liveness, valence, tempo)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    # Execute with parameter binding for all values
                    cursor.execute(sql, (
                        song_id, 
                        float(danceability),
                        float(energy),
                        float(key),
                        float(loudness),
                        float(mode),
                        float(speechiness),
                        float(acousticness),
                        float(instrumentalness),
                        float(liveness),
                        float(valence),
                        float(tempo)
                    ))
                    
                    total_features_loaded += 1
                    
                    # Commit every 10 records
                    if total_features_loaded % 10 == 0:
                        conn.commit()
                        
                except Exception as e:
                    logger.error(f"Error inserting audio features for {record.get('song_id', 'unknown')}: {str(e)}")
                    continue
            
            # Commit after each batch
            conn.commit()
            
            # Brief pause
            time.sleep(0.1)
        
        logger.info(f"Loaded {total_features_loaded} records into song_features table")
        
    except Exception as e:
        logger.error(f"Error during vector data loading: {str(e)}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
    
    # 6. Verify the data was loaded properly
    logger.info("Verifying data load...")
    verification_engine = create_engine(
        connection_string,
        connect_args={"ssl": {"ca": certifi.where()}, "ssl_verify_identity": True}
    )
    
    try:
        with verification_engine.connect() as connection:
            # Verify songs table
            result = connection.execute(text("SELECT COUNT(*) as count FROM songs")).fetchone()
            logger.info(f"Verified {result.count} records in songs table")
            
            # Verify vector table
            result = connection.execute(text("SELECT COUNT(*) as count FROM song_embeddings_vector")).fetchone()
            logger.info(f"Verified {result.count} records in song_embeddings_vector table")
            
            # Verify features table
            result = connection.execute(text("SELECT COUNT(*) as count FROM song_features")).fetchone()
            logger.info(f"Verified {result.count} records in song_features table")
            
            # Sample query to verify audio feature columns
            sample = connection.execute(text("SELECT danceability, energy, `key`, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo FROM song_features LIMIT 1")).fetchone()
            logger.info(f"Sample audio features: {sample}")
    except Exception as e:
        logger.error(f"Error verifying data: {str(e)}")
    
    logger.info(f"Successfully loaded songs into TiDB database with vector embeddings and individual audio features.")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise