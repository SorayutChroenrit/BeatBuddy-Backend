import json
import pandas as pd
import certifi
from sqlalchemy import create_engine, text, inspect
import pymysql
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Explicitly tell SQLAlchemy to use PyMySQL
pymysql.install_as_MySQLdb()

try:
    # 1. Load the processed JSON files
    logger.info("Loading JSON data...")
    
    # Load songs data
    try:
        # Try to read as a JSON file with multiple records
        songs_df = pd.read_json('data/db-kaggle/kaggle_processed_songs.json', orient='records')
        logger.info(f"Loaded {len(songs_df)} records from songs JSON using records orientation")
    except ValueError:
        try:
            # If the above fails, try reading as a regular JSON file
            songs_df = pd.read_json('data/db-kaggle/kaggle_processed_songs.json')
            logger.info(f"Loaded {len(songs_df)} records from songs JSON")
        except ValueError:
            # If still fails, try loading the file as a JSON Lines file (one JSON object per line)
            songs_df = pd.read_json('data/db-kaggle/kaggle_processed_songs.json', lines=True)
            logger.info(f"Loaded {len(songs_df)} records from songs JSON Lines file")
    
    # Map column names to match the expected format
    column_mapping = {
        'song': 'track_name',
        'artists': 'track_artist',
        'new_song_id': 'spotify_id',
    }
    
    # Rename columns according to mapping
    songs_df = songs_df.rename(columns=column_mapping)
    
    # Check for required columns and prepare DataFrames
    required_song_columns = ['song_id', 'track_name', 'track_artist']
    missing_columns = [col for col in required_song_columns if col not in songs_df.columns]
    if missing_columns:
        logger.error(f"Missing required columns in songs JSON: {missing_columns}")
        raise ValueError(f"Missing required columns in songs JSON: {missing_columns}")
    
    # Add this before processing records, right after loading the JSON
    sample_record = songs_df.iloc[0].to_dict()
    logger.info(f"Available fields in first record: {list(sample_record.keys())}")

    # Also check if 'cleaned_lyrics' exists as a column in the DataFrame
    logger.info(f"'cleaned_lyrics' exists as DataFrame column: {'cleaned_lyrics' in songs_df.columns}")
    # Create a list of song records from the DataFrame
    song_records = []
    for _, row in songs_df.iterrows():
        song_record = {
            'song_id': row['song_id'],
            'track_name': row['track_name'],
            'track_artist': row['track_artist'],
            'track_album_name': row.get('track_album_name', ''),
            'original_lyrics': row.get('Lyrics', row.get('original_lyrics', '')),
            'cleaned_lyrics': row.get('lyrics', ''), 
            'playlist_genre': row.get('playlist_genre', ''),
            'playlist_subgenre': row.get('playlist_subgenre', ''),
            'genres': row.get('genres', ''),
            'language': row.get('language', ''),
            'sentiment': row.get('sentiment', ''),
            'popularity_score': float(row.get('popularity', 0)),
            'link': row.get('link', ''),
            'spotify_id': row.get('spotify_id', ''),
            'siamzone_id': row.get('siamzone_id', '')
        }
        
        # Check for long links and truncate if needed
        max_link_length = 1000  # Slightly less than field size to be safe
        if song_record['link'] and len(str(song_record['link'])) > max_link_length:
            song_record['link'] = str(song_record['link'])[:max_link_length]
        
        song_records.append(song_record)
    
    logger.info(f"Processed {len(song_records)} song records with proper field mappings")
    
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

    # 3. Check for existing records to avoid duplicates
    with engine.connect() as connection:
        # Check if tables exist
        inspector = inspect(engine)
        required_tables = ['songs', 'song_embeddings_vector', 'song_features']
        missing_tables = [table for table in required_tables if table not in inspector.get_table_names()]
        
        if missing_tables:
            logger.error(f"Required tables don't exist: {missing_tables}")
            raise ValueError(f"Required tables don't exist in the database: {missing_tables}")
            
        # Get existing song IDs from the database
        existing_songs = pd.read_sql("SELECT song_id FROM songs", connection)
        existing_song_ids = set(existing_songs['song_id'].tolist())
        
        logger.info(f"Found {len(existing_song_ids)} existing songs in the database")
    
    # Filter out records that already exist in the database
    new_song_records = [record for record in song_records if record['song_id'] not in existing_song_ids]
    logger.info(f"Filtered to {len(new_song_records)} new songs to be inserted")
    
    # If no new records, exit early
    if len(new_song_records) == 0:
        logger.info("No new records to insert. Exiting.")
        exit(0)
    
    # 4. Load data into songs table
    logger.info("Loading new data into songs table...")
    batch_size = 50  # Smaller batch size for better transaction handling
    total_inserted = 0
    
    # Split records into smaller batches
    for batch_start in range(0, len(new_song_records), batch_size):
        batch_end = min(batch_start + batch_size, len(new_song_records))
        batch = new_song_records[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(new_song_records)-1)//batch_size + 1} (records {batch_start+1}-{batch_end})")
        
        # Create a new connection for each batch to avoid transaction issues
        batch_engine = create_engine(
            connection_string,
            connect_args={
                "ssl": {"ca": certifi.where()},
                "ssl_verify_identity": True,
                "connect_timeout": 60
            },
            isolation_level="READ COMMITTED"
        )
        
        # Convert batch to DataFrame for using to_sql
        batch_df = pd.DataFrame(batch)
        
        try:
            # Use with engine.begin() for automatic transaction management
            with batch_engine.begin() as connection:
                # Log sample record for debugging if this is the first batch
                if batch_start == 0:
                    logger.info(f"Sample record being inserted: {json.dumps(batch[0], default=str)}")
                
                # Insert batch
                batch_df.to_sql('songs', connection, if_exists='append', index=False)
                total_inserted += len(batch)
                logger.info(f"Inserted batch {batch_start//batch_size + 1} successfully")
        except Exception as e:
            logger.error(f"Error inserting batch {batch_start//batch_size + 1}: {str(e)}")
            continue  # Continue with next batch even if this one fails
        finally:
            # Dispose of the batch engine to close all connections
            batch_engine.dispose()
    
    logger.info(f"Successfully inserted {total_inserted} out of {len(new_song_records)} song records")
    
    # 5. Check for and load embeddings if they exist
    has_embeddings = False
    for record in new_song_records[:5]:  # Check first few records
        if 'lyrics_embedding' in record and 'metadata_embedding' in record:
            has_embeddings = True
            break
    
    if has_embeddings:
        logger.info("Found embedding data, preparing to load them...")
        
        # Create a direct PyMySQL connection for better control
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
            autocommit=False
        )
        
        try:
            cursor = conn.cursor()
            
            # Function to format vectors in a TiDB compatible format
            def format_vector_values(vector):
                if vector is None or len(vector) == 0:
                    return ""
                # Convert vector to list if it's not already
                if isinstance(vector, str):
                    try:
                        vector = json.loads(vector)
                    except:
                        logger.warning("Could not parse vector string, using empty vector")
                        return ""
                return ",".join([f"{v:.6f}" for v in vector])
            
            # Process embeddings in small batches
            embeddings_batch_size = 10  # Very small batch size for embeddings
            total_embeddings_inserted = 0
            
            for batch_start in range(0, len(new_song_records), embeddings_batch_size):
                batch_end = min(batch_start + embeddings_batch_size, len(new_song_records))
                batch = new_song_records[batch_start:batch_end]
                
                logger.info(f"Processing embeddings batch {batch_start//embeddings_batch_size + 1}/{(len(new_song_records)-1)//embeddings_batch_size + 1}")
                
                for record in batch:
                    try:
                        song_id = record['song_id']
                        
                        # Check if this record has embedding data
                        if 'lyrics_embedding' not in record or 'metadata_embedding' not in record:
                            continue
                        
                        lyrics_embedding = record.get('lyrics_embedding')
                        metadata_embedding = record.get('metadata_embedding')
                        
                        if not lyrics_embedding or not metadata_embedding:
                            logger.warning(f"Skipping {song_id} - missing embedding data")
                            continue
                        
                        # Format vectors as comma-separated strings
                        lyrics_values = format_vector_values(lyrics_embedding)
                        metadata_values = format_vector_values(metadata_embedding)
                        
                        # Create SQL with correct TiDB VECTOR syntax
                        sql = f"""
                        INSERT INTO song_embeddings_vector 
                        (song_id, lyrics_embedding, metadata_embedding)
                        VALUES (
                            %s, 
                            '[{lyrics_values}]', 
                            '[{metadata_values}]'
                        )
                        """
                        
                        # Execute with parameter binding only for song_id
                        cursor.execute(sql, (song_id,))
                        
                        total_embeddings_inserted += 1
                        
                        # Commit every 5 records to avoid transaction timeout
                        if total_embeddings_inserted % 5 == 0:
                            conn.commit()
                            
                    except Exception as e:
                        logger.error(f"Error inserting vector for {record.get('song_id', 'unknown')}: {str(e)}")
                        continue
                
                # Commit after each batch
                conn.commit()
                
                # Small pause to prevent overwhelming the database
                time.sleep(0.1)
            
            logger.info(f"Successfully inserted {total_embeddings_inserted} vector embeddings")
            
        except Exception as e:
            logger.error(f"Error during vector data loading: {str(e)}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    else:
        logger.info("No embedding data found in records")
    
    # 6. Verify the loaded data
    logger.info("Verifying data load...")
    with engine.connect() as connection:
        try:
            # Get counts from each table
            songs_count = connection.execute(text("SELECT COUNT(*) as count FROM songs")).scalar()
            logger.info(f"Total songs in database: {songs_count}")
            
            if has_embeddings:
                vector_count = connection.execute(text("SELECT COUNT(*) as count FROM song_embeddings_vector")).scalar()
                logger.info(f"Total vector embeddings in database: {vector_count}")
                
            # Sample verification query
            if total_inserted > 0:
                sample_id = new_song_records[0]['song_id']
                verification = connection.execute(text(f"SELECT * FROM songs WHERE song_id = '{sample_id}'")).fetchone()
                logger.info(f"Verified record exists for sample song_id: {sample_id}")
                
        except Exception as e:
            logger.error(f"Error verifying data: {str(e)}")

    logger.info(f"Data loading process completed successfully")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise