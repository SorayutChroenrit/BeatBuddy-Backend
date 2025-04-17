import json
import pandas as pd
import certifi
from sqlalchemy import create_engine, text, inspect
import pymysql
import logging
import ijson  # For streaming JSON parsing
import os
import decimal  # Import for Decimal handling

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Explicitly tell SQLAlchemy to use PyMySQL
pymysql.install_as_MySQLdb()

# Custom JSON encoder to handle Decimal values
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

try:
    # 1. Check if the JSON file is too large
    json_file_path = 'data/db-th-siamzone/processed_thai_songs.json'
    file_size_mb = os.path.getsize(json_file_path) / (1024 * 1024)
    logger.info(f"JSON file size: {file_size_mb:.2f} MB")
    
    # If the file is very large, we'll need to process it in chunks
    large_file = file_size_mb > 100  # Consider files over 100MB as large
    
    if large_file:
        logger.info("Large JSON file detected, using streaming approach...")
        
        # First, get a count of records
        record_count = 0
        with open(json_file_path, 'r', encoding='utf-8') as f:
            # Count objects in the JSON array
            for _ in ijson.items(f, 'item'):
                record_count += 1
        
        logger.info(f"Found {record_count} records in JSON file")
        
        # We'll process in batches
        batch_size = 1000
        
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
        
        # 3. Check if we need to add the Thai-specific columns to our database
        with engine.connect() as connection:
            inspector = inspect(engine)
            
            if 'songs' in inspector.get_table_names():
                # Get existing columns in songs table
                songs_table_columns = [col['name'] for col in inspector.get_columns('songs')]
                logger.info(f"Songs table exists with columns: {songs_table_columns}")
                
                # Check if we need to add the Thai-specific columns
                if 'tokenized_lyrics' not in songs_table_columns:
                    logger.info("Adding 'tokenized_lyrics' column to songs table...")
                    connection.execute(text("ALTER TABLE songs ADD COLUMN tokenized_lyrics TEXT AFTER cleaned_lyrics"))
                
                if 'filtered_lyrics' not in songs_table_columns:
                    logger.info("Adding 'filtered_lyrics' column to songs table...")
                    connection.execute(text("ALTER TABLE songs ADD COLUMN filtered_lyrics TEXT AFTER tokenized_lyrics"))
                    
                # Get all existing song_ids from the database
                existing_ids_query = text("SELECT song_id FROM songs")
                existing_ids_result = connection.execute(existing_ids_query).fetchall()
                existing_ids = set([row[0] for row in existing_ids_result])
                
                logger.info(f"Found {len(existing_ids)} existing records in the database")
            else:
                logger.error("Songs table doesn't exist in the database. Please create it first.")
                raise ValueError("Songs table doesn't exist")
        
        # Process in batches
        total_inserted = 0
        total_skipped = 0
        batch_num = 0
        
        # Process song records
        with open(json_file_path, 'r', encoding='utf-8') as f:
            # Read records in batches
            records_batch = []
            embeddings_batch = []
            
            for record in ijson.items(f, 'item'):
                if record['song_id'] not in existing_ids:
                    # Process for songs table
                    song_record = {
                        'song_id': record['song_id'],
                        'track_name': record.get('track_name', ''),
                        'track_artist': record.get('track_artist', ''),
                        'original_lyrics': record.get('original_lyrics', ''),
                        'cleaned_lyrics': record.get('cleaned_lyrics', ''),
                        'language': 'thai'
                    }
                    
                    # Add Thai-specific fields if they exist
                    if 'tokenized_lyrics' in record:
                        song_record['tokenized_lyrics'] = record['tokenized_lyrics']
                    if 'filtered_lyrics' in record:
                        song_record['filtered_lyrics'] = record['filtered_lyrics']
                    
                    # Add other fields with defaults if they're expected in the database
                    for col in songs_table_columns:
                        if col not in song_record and col not in ['song_id', 'track_name', 'track_artist', 
                                                                 'original_lyrics', 'cleaned_lyrics', 
                                                                 'tokenized_lyrics', 'filtered_lyrics',
                                                                 'language']:
                            if col == 'popularity_score':
                                song_record[col] = 0.0
                            else:
                                song_record[col] = ''
                    
                    records_batch.append(song_record)
                    
                    # Process for embeddings table if they exist
                    if 'lyrics_embedding' in record and 'metadata_embedding' in record:
                        embedding_record = {
                            'song_id': record['song_id'],
                            'lyrics_embedding': json.dumps(record['lyrics_embedding'], cls=DecimalEncoder),  # Use custom encoder
                            'metadata_embedding': json.dumps(record['metadata_embedding'], cls=DecimalEncoder)  # Use custom encoder
                        }
                        embeddings_batch.append(embedding_record)
                else:
                    total_skipped += 1
                
                # Insert batch when it reaches the specified size
                if len(records_batch) >= batch_size:
                    batch_num += 1
                    logger.info(f"Processing batch {batch_num} with {len(records_batch)} records")
                    
                    # Insert songs
                    if records_batch:
                        songs_df = pd.DataFrame(records_batch)
                        songs_df.to_sql('songs', engine, if_exists='append', index=False, chunksize=100)
                        total_inserted += len(records_batch)
                    
                    # Insert embeddings
                    if embeddings_batch:
                        embeddings_df = pd.DataFrame(embeddings_batch)
                        embeddings_df.to_sql('song_embeddings', engine, if_exists='append', index=False, chunksize=100)
                    
                    # Clear batches
                    records_batch = []
                    embeddings_batch = []
            
            # Insert any remaining records
            if records_batch:
                batch_num += 1
                logger.info(f"Processing final batch {batch_num} with {len(records_batch)} records")
                
                # Insert songs
                songs_df = pd.DataFrame(records_batch)
                songs_df.to_sql('songs', engine, if_exists='append', index=False, chunksize=100)
                total_inserted += len(records_batch)
                
                # Insert embeddings
                if embeddings_batch:
                    embeddings_df = pd.DataFrame(embeddings_batch)
                    embeddings_df.to_sql('song_embeddings', engine, if_exists='append', index=False, chunksize=100)
        
        logger.info(f"Completed processing: {total_inserted} records inserted, {total_skipped} records skipped (already exist)")
    
    else:
        # For smaller files, we can load everything at once
        logger.info("Loading JSON data...")
        
        # Load song data from JSON
        with open(json_file_path, 'r', encoding='utf-8') as f:
            song_records = json.load(f)
        
        # Convert JSON to DataFrame
        songs_df = pd.DataFrame(song_records)
        
        # Extract embeddings data if it's in the same JSON
        if 'lyrics_embedding' in songs_df.columns and 'metadata_embedding' in songs_df.columns:
            embeddings_df = pd.DataFrame({
                'song_id': songs_df['song_id'],
                'lyrics_embedding': songs_df['lyrics_embedding'].apply(lambda x: json.dumps(x, cls=DecimalEncoder) if not isinstance(x, str) else x),
                'metadata_embedding': songs_df['metadata_embedding'].apply(lambda x: json.dumps(x, cls=DecimalEncoder) if not isinstance(x, str) else x)
            })
        else:
            # Try to load from a CSV file if embeddings aren't in the JSON
            try:
                embeddings_df = pd.read_csv('data/db-th-siamzone/thai_song_embeddings.csv')
            except FileNotFoundError:
                logger.warning("Could not find embeddings CSV file")
                embeddings_df = pd.DataFrame(columns=['song_id', 'lyrics_embedding', 'metadata_embedding'])
            
        logger.info(f"Loaded {len(songs_df)} song records and {len(embeddings_df)} embedding records")

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

        # 3. Check if we need to add the Thai-specific columns to our database
        with engine.connect() as connection:
            inspector = inspect(engine)
            
            if 'songs' in inspector.get_table_names():
                # Get existing columns in songs table
                songs_table_columns = [col['name'] for col in inspector.get_columns('songs')]
                logger.info(f"Songs table exists with columns: {songs_table_columns}")
                
                # Check if we need to add the Thai-specific columns
                if 'tokenized_lyrics' not in songs_table_columns and 'tokenized_lyrics' in songs_df.columns:
                    logger.info("Adding 'tokenized_lyrics' column to songs table...")
                    connection.execute(text("ALTER TABLE songs ADD COLUMN tokenized_lyrics TEXT AFTER cleaned_lyrics"))
                
                if 'filtered_lyrics' not in songs_table_columns and 'filtered_lyrics' in songs_df.columns:
                    logger.info("Adding 'filtered_lyrics' column to songs table...")
                    connection.execute(text("ALTER TABLE songs ADD COLUMN filtered_lyrics TEXT AFTER tokenized_lyrics"))
                    
                # Update columns list after potential additions
                songs_table_columns = [col['name'] for col in inspector.get_columns('songs')]
                
                # Create a new processed DataFrame that matches the database structure
                processed_songs_df = pd.DataFrame()
                
                # Map the DataFrame columns to match the database columns
                for db_col in songs_table_columns:
                    if db_col in songs_df.columns:
                        processed_songs_df[db_col] = songs_df[db_col]
                    elif db_col == 'cleaned_lyrics' and 'lyrics' in songs_df.columns:
                        # Map 'lyrics' to 'cleaned_lyrics' if needed
                        processed_songs_df['cleaned_lyrics'] = songs_df['lyrics']
                    else:
                        # Set appropriate default values based on column name
                        if db_col in ['track_album_name', 'playlist_genre', 'playlist_subgenre', 'genres', 'sentiment', 'link', 'spotify_id']:
                            processed_songs_df[db_col] = ''
                        elif db_col == 'language':
                            processed_songs_df[db_col] = 'thai'  # Set language for Thai songs
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
            
        # Filter out records that already exist in the database
        new_songs = processed_songs_df[~processed_songs_df['song_id'].isin(existing_ids)]
        new_embeddings = embeddings_df[~embeddings_df['song_id'].isin(existing_ids)]
        
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
                # Process in smaller batches to avoid memory issues
                batch_size = 500
                total_batches = len(new_songs) // batch_size + (1 if len(new_songs) % batch_size > 0 else 0)
                
                for i in range(total_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(new_songs))
                    batch = new_songs.iloc[start_idx:end_idx]
                    
                    logger.info(f"Loading songs batch {i+1}/{total_batches} (records {start_idx+1}-{end_idx})")
                    batch.to_sql('songs', engine, if_exists='append', index=False, chunksize=chunksize)
                
                logger.info(f"Successfully loaded {len(new_songs)} new records into songs table")
            except Exception as e:
                logger.error(f"Error loading songs table: {str(e)}")
                raise
            
            # Reset connection for the next table load
            engine.dispose()
            
            # 6. Load embeddings data
            if len(new_embeddings) > 0:
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

    logger.info("Database operation completed successfully.")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise