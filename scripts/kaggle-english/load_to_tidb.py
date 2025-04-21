import json
import pandas as pd
import certifi
import pymysql
import logging
import time
import uuid  # For generating new song IDs

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # 1. Load the JSON file
    logger.info("Loading JSON data...")
    songs_df = pd.read_json('data/embedding/kaggle_embeddings.json', orient='records')
    logger.info(f"Loaded {len(songs_df)} records from JSON file")
    
    # 2. Map column names if needed
    column_mapping = {
        'song': 'track_name',
        'artists': 'track_artist',
        'new_song_id': 'spotify_id',
    }
    songs_df = songs_df.rename(columns=column_mapping)
    
    # 3. Check for embedding columns
    # Mapping between our column names and database column names
    embedding_mapping = {
        'lyrics_embedding': 'lyrics_embedding',
        'track_name_embedding': 'track_name_embedding',
        'track_artist_embedding': 'artist_embedding',  
        'genres_embedding': 'genres_embedding'
    }
    
    embedding_types = ['lyrics_embedding', 'track_name_embedding', 'track_artist_embedding', 'genres_embedding']
    existing_embedding_types = [col for col in embedding_types if col in songs_df.columns]
    
    if not existing_embedding_types:
        logger.error("No embedding columns found in the data")
        raise ValueError("Missing embedding data")
        
    logger.info(f"Found embedding columns: {existing_embedding_types}")
    
    # 4. Get embedding dimension
    vector_dimension = None
    for emb_type in existing_embedding_types:
        if not songs_df[emb_type].isna().all():
            # Find first non-null embedding
            sample_embedding = None
            for _, row in songs_df.iterrows():
                # Check if the value exists and isn't None
                if row[emb_type] is not None and not pd.isna(row[emb_type]).all():
                    sample_embedding = row[emb_type]
                    break
                    
            if sample_embedding is not None:
                # Convert to list if it's a string
                if isinstance(sample_embedding, str):
                    try:
                        sample_embedding = json.loads(sample_embedding)
                    except:
                        pass
                        
                vector_dimension = len(sample_embedding)
                logger.info(f"Detected embedding dimension: {vector_dimension} from {emb_type}")
                break
    
    if vector_dimension is None:
        logger.error("Could not determine embedding dimension")
        raise ValueError("Invalid embedding data")
    
    # 5. Create direct connection to TiDB
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
    
    cursor = conn.cursor()
    
    try:
        # Get existing tables to check columns
        cursor.execute("SHOW TABLES LIKE 'songs'")
        songs_table_exists = cursor.fetchone() is not None
        
        if not songs_table_exists:
            logger.error("Songs table doesn't exist in the database. Please create it first.")
            raise ValueError("Songs table doesn't exist")
            
        # Get songs table columns
        cursor.execute("DESCRIBE songs")
        songs_table_columns = [row[0] for row in cursor.fetchall()]
        logger.info(f"Songs table columns: {songs_table_columns}")
        
        # Check if song_embeddings_vector table exists
        cursor.execute("SHOW TABLES LIKE 'song_embeddings_vector'")
        embeddings_table_exists = cursor.fetchone() is not None
        
        if embeddings_table_exists:
            # Get columns in embeddings table
            cursor.execute("DESCRIBE song_embeddings_vector")
            embeddings_table_columns = [row[0] for row in cursor.fetchall()]
            logger.info(f"Embeddings table columns: {embeddings_table_columns}")
            
            # Check which columns are NOT NULL
            not_null_columns = []
            cursor.execute("DESCRIBE song_embeddings_vector")
            for row in cursor.fetchall():
                col_name = row[0]
                is_null = row[2].upper()  # 'YES' or 'NO'
                if is_null == 'NO':
                    not_null_columns.append(col_name)
            
            logger.info(f"NOT NULL columns in embeddings table: {not_null_columns}")
        
        # 6. Process the records - directly insert new songs
        # Convert DataFrame to list of dicts for easier processing
        records = songs_df.to_dict('records')
        logger.info(f"Processing {len(records)} records for insertion")
        
        # Process in batches
        batch_size = 100
        total_songs_inserted = 0
        total_embeddings_inserted = 0
        
        # Default embedding for missing required columns
        default_embedding_json = json.dumps([0.0] * vector_dimension)
        
        # Process in batches
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(records) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} (records {i+1}-{min(i+batch_size, len(records))})")
            
            # Process each record
            for record in batch:
                try:
                    # Ensure each record has a song_id
                    if 'song_id' not in record or not record['song_id']:
                        record['song_id'] = str(uuid.uuid4())
                        
                    song_id = record['song_id']
                    logger.info(f"Processing record with song_id: {song_id}")
                    
                    # 1. Insert into songs table
                    song_record = {
                        'song_id': song_id,
                        'track_name': record.get('track_name', ''),
                        'track_artist': record.get('track_artist', ''),
                        'language': 'english'  # Default for Kaggle data
                    }
                    
                    # Add lyrics if present
                    if 'lyrics' in record and record['lyrics']:
                        song_record['original_lyrics'] = record['lyrics']
                        song_record['cleaned_lyrics'] = record['lyrics']
                    
                    # Add spotify_id if present
                    if 'spotify_id' in record and record['spotify_id']:
                        song_record['spotify_id'] = record['spotify_id']
                    
                    # Filter song_record to include only columns that exist in the songs table
                    filtered_song_record = {k: v for k, v in song_record.items() if k in songs_table_columns}
                    
                    # Insert into songs table
                    columns = list(filtered_song_record.keys())
                    placeholders = ["%s"] * len(columns)
                    
                    sql = f"""
                    INSERT INTO songs 
                    ({', '.join(columns)})
                    VALUES ({', '.join(placeholders)})
                    ON DUPLICATE KEY UPDATE song_id=song_id
                    """
                    
                    values = tuple(filtered_song_record.values())
                    cursor.execute(sql, values)
                    total_songs_inserted += 1
                    
                    # 2. Insert embeddings
                    # Prepare embeddings
                    embedding_record = {'song_id': song_id}
                    
                    # Process each embedding field
                    for json_col, db_col in embedding_mapping.items():
                        if json_col in record and record[json_col] is not None:
                            # Format as JSON
                            if isinstance(record[json_col], str):
                                try:
                                    vector = json.loads(record[json_col])
                                    embedding_record[db_col] = json.dumps(vector)
                                except:
                                    logger.warning(f"Could not parse {json_col} for {song_id}")
                            else:
                                embedding_record[db_col] = json.dumps(record[json_col])
                    
                    # Handle required fields
                    if embeddings_table_exists:
                        for col in not_null_columns:
                            if col != 'song_id' and col not in embedding_record:
                                logger.info(f"Adding default value for required column {col}")
                                embedding_record[col] = default_embedding_json
                    
                    # Filter embedding_record to include only columns that exist in the embeddings table
                    if embeddings_table_exists:
                        filtered_embedding_record = {k: v for k, v in embedding_record.items() if k in embeddings_table_columns}
                    else:
                        filtered_embedding_record = embedding_record
                    
                    # Insert into embeddings table
                    if len(filtered_embedding_record) > 1:  # More than just song_id
                        columns = list(filtered_embedding_record.keys())
                        placeholders = ["%s"] * len(columns)
                        
                        sql = f"""
                        INSERT INTO song_embeddings_vector 
                        ({', '.join(columns)})
                        VALUES ({', '.join(placeholders)})
                        ON DUPLICATE KEY UPDATE song_id=song_id
                        """
                        
                        values = tuple(filtered_embedding_record.values())
                        cursor.execute(sql, values)
                        total_embeddings_inserted += 1
                
                except Exception as e:
                    logger.error(f"Error processing record with song_id {record.get('song_id', 'unknown')}: {str(e)}")
                    continue
            
            # Commit after each batch
            conn.commit()
            logger.info(f"Batch {batch_num} completed - Inserted {total_songs_inserted} songs and {total_embeddings_inserted} embeddings so far")
        
        logger.info(f"Insertion completed: Added {total_songs_inserted} songs and {total_embeddings_inserted} embeddings")
        
        # Verify results
        cursor.execute("SELECT COUNT(*) FROM songs")
        total_songs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM song_embeddings_vector")
        total_embeddings = cursor.fetchone()[0]
        
        logger.info(f"Final database counts: {total_songs} songs, {total_embeddings} songs with embeddings")
        
    except Exception as e:
        logger.error(f"Error during data insertion: {str(e)}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()
        
    logger.info("Data loading process completed")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise