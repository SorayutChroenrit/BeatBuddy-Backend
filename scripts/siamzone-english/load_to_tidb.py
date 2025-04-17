import json
import pandas as pd
import certifi
from sqlalchemy import create_engine, text, inspect
import pymysql
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Explicitly tell SQLAlchemy to use PyMySQL
pymysql.install_as_MySQLdb()

try:
    # 1. Load the processed CSV file
    logger.info("Loading CSV data...")
    
    # Load song data from CSV file - update the path to match your file location
    file_path = '/content/drive/MyDrive/dataset/processed/english_processed_songs.csv'
    logger.info(f"Reading CSV file from: {file_path}")
    
    # Load the CSV using pandas
    songs_df = pd.read_csv(file_path)
    
    # Show information about the loaded data
    logger.info(f"Loaded {len(songs_df)} records from CSV")
    logger.info(f"CSV columns: {songs_df.columns.tolist()}")
    
    # Check presence of required columns
    required_columns = ['Song ID', 'track_name', 'track_artist', 'Lyrics', 'original_lyrics', 'cleaned_lyrics']
    missing_columns = [col for col in required_columns if col not in songs_df.columns]
    
    if missing_columns:
        logger.warning(f"Missing required columns: {missing_columns}")
    else:
        logger.info("All required columns present in CSV")
    
    # Rename columns to match database schema if needed
    # 'Song ID' is actually the siamzone_id, not the primary key
    column_mapping = {
        'Song ID': 'siamzone_id',
        'Lyrics': 'original_lyrics'
    }
    
    # Make sure we're using the UUID field in 'song_id' as the primary key
    if 'song_id' in songs_df.columns:
        logger.info("Found 'song_id' (UUID) column to use as primary key.")
    else:
        logger.error("No 'song_id' (UUID) column found. This is required as the primary key.")
        raise ValueError("Missing song_id (UUID) column in the data")
    
    # Only apply renaming for columns that exist and need to be renamed
    columns_to_rename = {old: new for old, new in column_mapping.items() 
                        if old in songs_df.columns and old != new}
    
    if columns_to_rename:
        songs_df = songs_df.rename(columns=columns_to_rename)
        logger.info(f"Renamed columns: {columns_to_rename}")
    
    # Check for embedding columns
    has_lyrics_embedding = 'lyrics_embedding' in songs_df.columns
    has_metadata_embedding = 'metadata_embedding' in songs_df.columns
    
    if has_lyrics_embedding or has_metadata_embedding:
        logger.info("Embedding columns found in CSV")
        # Extract embedding columns
        embedding_columns = ['song_id']
        if has_lyrics_embedding:
            embedding_columns.append('lyrics_embedding')
        if has_metadata_embedding:
            embedding_columns.append('metadata_embedding')
            
        # Create embeddings DataFrame
        embeddings_df = songs_df[embedding_columns].copy()
        logger.info(f"Created embeddings DataFrame with {len(embeddings_df)} records")
    else:
        logger.info("No embedding columns found in CSV")
        embeddings_df = pd.DataFrame(columns=['song_id'])

    # 2. TiDB Connection Configuration
    connection_string = "mysql+pymysql://TpjWXdSdsSyb73z.root:xk0bNXkyhjg9X5xb@gateway01.ap-southeast-1.prod.aws.tidbcloud.com:4000/music_recommendation"
    
    # Create engine with SSL configuration
    logger.info("Connecting to TiDB with SQLAlchemy + PyMySQL...")
    engine = create_engine(
        connection_string,
        connect_args={
            "ssl": {"ca": certifi.where()},
            "ssl_verify_identity": True
        },
        isolation_level="READ COMMITTED"
    )

    # 3. Get database table structure and adapt our data to it
    with engine.connect() as connection:
        # Using inspector to get table columns
        inspector = inspect(engine)
        
        if 'songs' in inspector.get_table_names():
            # Get existing columns in songs table
            songs_table_columns = [col['name'] for col in inspector.get_columns('songs')]
            logger.info(f"Songs table exists with columns: {songs_table_columns}")
            
            # Create a new processed DataFrame that matches the database structure
            processed_songs_df = pd.DataFrame()
            
            # Map the DataFrame columns to match the database columns
            for db_col in songs_table_columns:
                if db_col in songs_df.columns:
                    # Direct mapping where column names match
                    processed_songs_df[db_col] = songs_df[db_col]
                
                # Special case mappings
                elif db_col == 'cleaned_lyrics' and 'lyrics' in songs_df.columns:
                    # Map 'lyrics' to 'cleaned_lyrics' if that's how your data is structured
                    processed_songs_df[db_col] = songs_df['lyrics']
                
                # Set default values for missing columns
                else:
                    if db_col in ['track_album_name', 'playlist_genre', 'playlist_subgenre', 'genres', 'language', 'sentiment', 'link', 'spotify_id', 'siamzone_id']:
                        processed_songs_df[db_col] = ''
                    elif db_col == 'popularity_score':
                        processed_songs_df[db_col] = 0.0
                    else:
                        processed_songs_df[db_col] = None
                            
            logger.info(f"Processed songs DataFrame to match database schema. Columns: {processed_songs_df.columns.tolist()}")
        else:
            logger.error("Songs table doesn't exist in the database. Please create it first.")
            raise ValueError("Songs table doesn't exist")

    # 4. Check for existing records to avoid duplicates
    logger.info("Checking for existing records to avoid duplicates...")
    
    # Get all existing song_ids from the database
    with engine.connect() as conn:
        existing_ids_query = text("SELECT song_id FROM songs")
        existing_ids_result = conn.execute(existing_ids_query).fetchall()
        existing_ids = set([row[0] for row in existing_ids_result])
    
    # Log a sample of UUIDs to verify format    
    if len(processed_songs_df) > 0:
        logger.info(f"Sample song_id (UUID) from data: {processed_songs_df['song_id'].iloc[0]}")
    
    # Filter out records that already exist in the database
    new_songs = processed_songs_df[~processed_songs_df['song_id'].isin(existing_ids)]
    
    # Only filter embeddings if we have any
    if len(embeddings_df) > 0 and 'song_id' in embeddings_df.columns:
        new_embeddings = embeddings_df[~embeddings_df['song_id'].isin(existing_ids)]
    else:
        new_embeddings = embeddings_df
    
    logger.info(f"Found {len(existing_ids)} existing records in the database")
    logger.info(f"Will insert {len(new_songs)} new records (skipping {len(processed_songs_df) - len(new_songs)} duplicates)")
    
    # If there are no new records, we're done
    if len(new_songs) == 0:
        logger.info("No new records to insert. All records already exist in the database.")
    else:
        # 5. Load new data into songs table
        logger.info("Loading new data into songs table...")
        chunksize = 100  # Smaller chunks for better transaction handling
        
        try:
            new_songs.to_sql('songs', engine, if_exists='append', index=False, chunksize=chunksize)
            logger.info(f"Successfully loaded {len(new_songs)} new records into songs table")
        except Exception as e:
            logger.error(f"Error loading songs table: {str(e)}")
            raise
        
        # Reset connection for the next table load
        engine.dispose()
        
        # 6. Load embeddings data (if there are any embeddings with actual data)
        if len(new_embeddings) > 0 and ('lyrics_embedding' in new_embeddings.columns or 'metadata_embedding' in new_embeddings.columns):
            logger.info("Loading embeddings data...")
            embedding_engine = create_engine(
                connection_string,
                connect_args={
                    "ssl": {"ca": certifi.where()},
                    "ssl_verify_identity": True
                },
                isolation_level="READ COMMITTED"
            )
            
            try:
                # Process in smaller batches to avoid memory issues
                batch_size = 500
                total_batches = len(new_embeddings) // batch_size + (1 if len(new_embeddings) % batch_size > 0 else 0)
                
                for i in range(total_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(new_embeddings))
                    batch = new_embeddings.iloc[start_idx:end_idx]
                    
                    logger.info(f"Loading embeddings batch {i+1}/{total_batches} (records {start_idx+1}-{end_idx})")
                    batch.to_sql('song_embeddings', embedding_engine, if_exists='append', index=False, chunksize=50)
                    
                logger.info(f"Loaded all {len(new_embeddings)} records into song_embeddings table")
            except Exception as e:
                logger.error(f"Error loading song_embeddings table: {str(e)}")
                raise
        else:
            logger.info("No embedding data to insert. Skipping embeddings table.")

    logger.info("Database operation completed successfully.")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise