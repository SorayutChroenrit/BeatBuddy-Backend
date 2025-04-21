import json
import pandas as pd
import certifi
import pymysql
import logging
import time
import ijson  # For streaming JSON parsing
import os
import decimal  # Import for Decimal handling

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle Decimal values
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

try:
    # 1. Check if the JSON file is too large
    json_file_path = 'data/embedding/siamzoneth_embedding.json'  # Thai version
    file_size_mb = os.path.getsize(json_file_path) / (1024 * 1024)
    logger.info(f"JSON file size: {file_size_mb:.2f} MB")
    
    # Define embedding column mapping (JSON column name -> DB column name)
    embedding_column_mapping = {
        'lyrics_embedding': 'lyrics_embedding',
        'track_name_embedding': 'track_name_embedding',
        'track_artist_embedding': 'artist_embedding',  # Map to artist_embedding in DB
        'genres_embedding': 'genres_embedding'
    }
    
    logger.info(f"Embedding column mapping defined: {embedding_column_mapping}")
    
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
        
        # 2. Create direct connection to TiDB
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
            # 3. Check if songs table exists
            cursor.execute("SHOW TABLES LIKE 'songs'")
            songs_table_exists = cursor.fetchone() is not None
            
            if songs_table_exists:
                # Get existing columns in songs table
                cursor.execute("DESCRIBE songs")
                songs_table_columns = [row[0] for row in cursor.fetchall()]
                logger.info(f"Songs table exists with columns: {songs_table_columns}")
                
                # Check if song_embeddings_vector table exists
                cursor.execute("SHOW TABLES LIKE 'song_embeddings_vector'")
                embeddings_table_exists = cursor.fetchone() is not None
                
                if not embeddings_table_exists:
                    logger.warning("song_embeddings_vector table doesn't exist. Will attempt to create it.")
                    # Create song_embeddings_vector table with correct column names
                    create_embeddings_table_query = """
                    CREATE TABLE IF NOT EXISTS song_embeddings_vector (
                        song_id VARCHAR(36) PRIMARY KEY,
                        lyrics_embedding JSON,
                        track_name_embedding JSON,
                        artist_embedding JSON,
                        genres_embedding JSON,
                        FOREIGN KEY (song_id) REFERENCES songs(song_id)
                    )
                    """
                    cursor.execute(create_embeddings_table_query)
                    conn.commit()
                    logger.info("Created song_embeddings_vector table with correct column names.")
                else:
                    # Check actual columns in the embeddings table
                    cursor.execute("DESCRIBE song_embeddings_vector")
                    embeddings_table_columns = [row[0] for row in cursor.fetchall()]
                    logger.info(f"song_embeddings_vector table exists with columns: {embeddings_table_columns}")
                
                # Get all existing song_ids from the database
                cursor.execute("SELECT song_id FROM songs")
                existing_ids_result = cursor.fetchall()
                existing_ids = set([row[0] for row in existing_ids_result])
                
                logger.info(f"Found {len(existing_ids)} existing records in the database")
            else:
                logger.error("Songs table doesn't exist in the database. Please create it first.")
                raise ValueError("Songs table doesn't exist")
            
            # Process in batches
            total_inserted = 0
            total_skipped = 0
            batch_num = 0
            
            # Track embedding statistics
            embedding_stats = {col: 0 for col in embedding_column_mapping.keys()}
            total_records_processed = 0
            
            # Process song records
            with open(json_file_path, 'r', encoding='utf-8') as f:
                # First, peek at a single record to log its columns
                first_record = next(ijson.items(f, 'item'), None)
                if first_record:
                    logger.info(f"First record column names: {list(first_record.keys())}")
                    
                    # Check each embedding column
                    for json_col in embedding_column_mapping.keys():
                        if json_col in first_record:
                            logger.info(f"Column '{json_col}' is present in first record")
                        else:
                            logger.warning(f"Column '{json_col}' is missing in first record")
                
                # Reset file pointer to beginning
                f.seek(0)
                
                # Read records in batches
                records_batch = []
                embeddings_batch = []
                
                for record in ijson.items(f, 'item'):
                    total_records_processed += 1
                    
                    # Track embedding column presence
                    for col in embedding_column_mapping.keys():
                        if col in record and record[col] is not None:
                            embedding_stats[col] += 1
                    
                    if record['song_id'] not in existing_ids:
                        # Process for songs table
                        song_record = {
                            'song_id': record['song_id'],
                            'track_name': record.get('track_name', ''),
                            'track_artist': record.get('track_artist', ''),
                            'original_lyrics': record.get('original_lyrics', ''),
                            'cleaned_lyrics': record.get('cleaned_lyrics', record.get('lyrics', '')),
                            'language': 'thai'  # Set to Thai
                        }
                        
                        # Add siamzone_id if present
                        if 'siamzone_id' in record:
                            song_record['siamzone_id'] = record['siamzone_id']
                        
                        # Add record to batch
                        records_batch.append(song_record)
                        
                        # Process for embeddings table
                        embedding_record = {'song_id': record['song_id']}
                        
                        # Add each available embedding field using the mapping
                        for json_col, db_col in embedding_column_mapping.items():
                            if json_col in record and record[json_col] is not None:
                                embedding_record[db_col] = json.dumps(record[json_col], cls=DecimalEncoder)
                        
                        # Only add to batch if it has at least one embedding
                        if len(embedding_record) > 1:  # More than just 'song_id'
                            embeddings_batch.append(embedding_record)
                    else:
                        total_skipped += 1
                    
                    # Log embedding stats every 10,000 records
                    if total_records_processed % 10000 == 0:
                        logger.info(f"Processed {total_records_processed} records so far")
                        for col, count in embedding_stats.items():
                            logger.info(f"Column '{col}' present in {count} records ({count/total_records_processed*100:.2f}%)")
                    
                    # Insert batch when it reaches the specified size
                    if len(records_batch) >= batch_size:
                        batch_num += 1
                        logger.info(f"Processing batch {batch_num} with {len(records_batch)} records")
                        
                        # Log the columns of the first record in this batch for verification
                        if records_batch:
                            logger.info(f"Sample record columns in batch {batch_num}: {list(records_batch[0].keys())}")
                        
                        # Insert songs using direct SQL execution
                        if records_batch:
                            for song in records_batch:
                                columns = list(song.keys())
                                placeholders = ["%s"] * len(columns)
                                
                                sql = f"""
                                INSERT INTO songs 
                                ({', '.join(columns)})
                                VALUES ({', '.join(placeholders)})
                                """
                                
                                values = tuple(song.values())
                                try:
                                    cursor.execute(sql, values)
                                except Exception as e:
                                    logger.error(f"Error inserting song {song['song_id']}: {str(e)}")
                            
                            total_inserted += len(records_batch)
                        
                        # Insert embeddings
                        if embeddings_batch:
                            # Log the columns of the first embedding in this batch
                            logger.info(f"Sample embedding columns in batch {batch_num}: {list(embeddings_batch[0].keys())}")
                            
                            # Count embeddings containing each column
                            embedding_column_counts = {col: 0 for col in embedding_column_mapping.values()}
                            for emb in embeddings_batch:
                                for col in emb.keys():
                                    if col != 'song_id' and col in embedding_column_counts:
                                        embedding_column_counts[col] += 1
                            
                            # Log the counts
                            for col, count in embedding_column_counts.items():
                                logger.info(f"Column '{col}' present in {count}/{len(embeddings_batch)} embeddings in this batch")
                            
                            for embedding in embeddings_batch:
                                columns = list(embedding.keys())
                                placeholders = ["%s"] * len(columns)
                                
                                sql = f"""
                                INSERT INTO song_embeddings_vector 
                                ({', '.join(columns)})
                                VALUES ({', '.join(placeholders)})
                                """
                                
                                values = tuple(embedding.values())
                                try:
                                    cursor.execute(sql, values)
                                except Exception as e:
                                    logger.error(f"Error inserting embedding for {embedding['song_id']}: {str(e)}")
                            
                            logger.info(f"Inserted {len(embeddings_batch)} embedding records")
                        
                        # Commit after each batch
                        conn.commit()
                        
                        # Clear batches
                        records_batch = []
                        embeddings_batch = []
                
                # Insert any remaining records
                if records_batch:
                    batch_num += 1
                    logger.info(f"Processing final batch {batch_num} with {len(records_batch)} records")
                    
                    # Log the columns of the first record in the final batch for verification
                    logger.info(f"Sample record columns in final batch: {list(records_batch[0].keys())}")
                    
                    # Insert songs
                    for song in records_batch:
                        columns = list(song.keys())
                        placeholders = ["%s"] * len(columns)
                        
                        sql = f"""
                        INSERT INTO songs 
                        ({', '.join(columns)})
                        VALUES ({', '.join(placeholders)})
                        """
                        
                        values = tuple(song.values())
                        try:
                            cursor.execute(sql, values)
                        except Exception as e:
                            logger.error(f"Error inserting song {song['song_id']}: {str(e)}")
                    
                    total_inserted += len(records_batch)
                    
                    # Insert embeddings
                    if embeddings_batch:
                        # Log the columns of the first embedding in the final batch
                        logger.info(f"Sample embedding columns in final batch: {list(embeddings_batch[0].keys())}")
                        
                        for embedding in embeddings_batch:
                            columns = list(embedding.keys())
                            placeholders = ["%s"] * len(columns)
                            
                            sql = f"""
                            INSERT INTO song_embeddings_vector 
                            ({', '.join(columns)})
                            VALUES ({', '.join(placeholders)})
                            """
                            
                            values = tuple(embedding.values())
                            try:
                                cursor.execute(sql, values)
                            except Exception as e:
                                logger.error(f"Error inserting embedding for {embedding['song_id']}: {str(e)}")
                        
                        logger.info(f"Inserted {len(embeddings_batch)} embedding records in final batch")
                    
                    conn.commit()
            
            # Log final embedding statistics
            logger.info("Final embedding column statistics:")
            for col, count in embedding_stats.items():
                logger.info(f"Column '{col}' present in {count}/{total_records_processed} records ({count/total_records_processed*100:.2f}%)")
            
            logger.info(f"Completed processing: {total_inserted} records inserted, {total_skipped} records skipped (already exist)")
        
        except Exception as e:
            logger.error(f"Error during data loading: {str(e)}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    else:
        # For smaller files, we can load everything at once
        logger.info("Loading JSON data...")
        
        # Load song data from JSON (not CSV!)
        with open(json_file_path, 'r', encoding='utf-8') as f:
            song_records = json.load(f)
            
        # Log columns from the first record if available
        if song_records and len(song_records) > 0:
            logger.info(f"JSON record columns: {list(song_records[0].keys())}")
            
            # Check for missing embedding columns
            for json_col in embedding_column_mapping.keys():
                if json_col not in song_records[0]:
                    logger.warning(f"Column '{json_col}' is missing in first record")
                else:
                    logger.info(f"Column '{json_col}' is present in first record")
        
        # Create direct connection to TiDB
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
            # Check if songs table exists
            cursor.execute("SHOW TABLES LIKE 'songs'")
            songs_table_exists = cursor.fetchone() is not None
            
            if songs_table_exists:
                # Get existing columns in songs table
                cursor.execute("DESCRIBE songs")
                songs_table_columns = [row[0] for row in cursor.fetchall()]
                logger.info(f"Songs table exists with columns: {songs_table_columns}")
                
                # Check if song_embeddings_vector table exists
                cursor.execute("SHOW TABLES LIKE 'song_embeddings_vector'")
                embeddings_table_exists = cursor.fetchone() is not None
                
                if not embeddings_table_exists:
                    logger.warning("song_embeddings_vector table doesn't exist. Will attempt to create it.")
                    # Create song_embeddings_vector table with correct column names
                    create_embeddings_table_query = """
                    CREATE TABLE IF NOT EXISTS song_embeddings_vector (
                        song_id VARCHAR(36) PRIMARY KEY,
                        lyrics_embedding JSON,
                        track_name_embedding JSON,
                        artist_embedding JSON,
                        genres_embedding JSON,
                        FOREIGN KEY (song_id) REFERENCES songs(song_id)
                    )
                    """
                    cursor.execute(create_embeddings_table_query)
                    conn.commit()
                    logger.info("Created song_embeddings_vector table with correct column names.")
                else:
                    # Check actual columns in the embeddings table
                    cursor.execute("DESCRIBE song_embeddings_vector")
                    embeddings_table_columns = [row[0] for row in cursor.fetchall()]
                    logger.info(f"song_embeddings_vector table exists with columns: {embeddings_table_columns}")
                
                # Get all existing song_ids from the database
                cursor.execute("SELECT song_id FROM songs")
                existing_ids_result = cursor.fetchall()
                existing_ids = set([row[0] for row in existing_ids_result])
                
                logger.info(f"Found {len(existing_ids)} existing records in the database")
                
                # Filter out records that already exist in the database
                new_songs = [song for song in song_records if song['song_id'] not in existing_ids]
                logger.info(f"Will insert {len(new_songs)} new records (skipping {len(song_records) - len(new_songs)} duplicates)")
                
                # Count embedding column presence
                embedding_stats = {col: 0 for col in embedding_column_mapping.keys()}
                for song in song_records:
                    for col in embedding_column_mapping.keys():
                        if col in song and song[col] is not None:
                            embedding_stats[col] += 1
                
                # Log embedding statistics
                logger.info("Embedding column statistics:")
                for col, count in embedding_stats.items():
                    logger.info(f"Column '{col}' present in {count}/{len(song_records)} records ({count/len(song_records)*100:.2f}%)")
                
                # Process in smaller batches
                batch_size = 100
                total_batches = len(new_songs) // batch_size + (1 if len(new_songs) % batch_size > 0 else 0)
                
                for i in range(total_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(new_songs))
                    batch = new_songs[start_idx:end_idx]
                    
                    logger.info(f"Processing batch {i+1}/{total_batches} (records {start_idx+1}-{end_idx})")
                    
                    # Log sample record from this batch
                    if batch:
                        logger.info(f"Sample record in batch {i+1}: song_id={batch[0]['song_id']}, columns={list(batch[0].keys())}")
                    
                    # Insert songs
                    for song in batch:
                        # Create a song record with essential fields
                        song_record = {
                            'song_id': song['song_id'],
                            'track_name': song.get('track_name', ''),
                            'track_artist': song.get('track_artist', ''),
                            'original_lyrics': song.get('original_lyrics', ''),
                            'cleaned_lyrics': song.get('cleaned_lyrics', song.get('lyrics', '')),
                            'language': 'thai'  # Set to Thai
                        }
                        
                        # Add siamzone_id if present
                        if 'siamzone_id' in song:
                            song_record['siamzone_id'] = song['siamzone_id']
                        
                        columns = list(song_record.keys())
                        placeholders = ["%s"] * len(columns)
                        
                        sql = f"""
                        INSERT INTO songs 
                        ({', '.join(columns)})
                        VALUES ({', '.join(placeholders)})
                        """
                        
                        values = tuple(song_record.values())
                        try:
                            cursor.execute(sql, values)
                        except Exception as e:
                            logger.error(f"Error inserting song {song['song_id']}: {str(e)}")
                    
                    # Insert embeddings
                    embedding_count = 0
                    # Count columns in this batch
                    embedding_column_counts = {col: 0 for col in embedding_column_mapping.values()}
                    
                    for song in batch:
                        embedding_record = {'song_id': song['song_id']}
                        
                        # Add each available embedding field using the mapping
                        for json_col, db_col in embedding_column_mapping.items():
                            if json_col in song and song[json_col] is not None:
                                embedding_record[db_col] = json.dumps(song[json_col], cls=DecimalEncoder)
                                embedding_column_counts[db_col] += 1
                        
                        # Only insert if it has at least one embedding
                        if len(embedding_record) > 1:  # More than just 'song_id'
                            # Log the first embedding record in each batch
                            if embedding_count == 0:
                                logger.info(f"Sample embedding in batch {i+1}: song_id={embedding_record['song_id']}, columns={list(embedding_record.keys())}")
                                embedding_count += 1
                                
                            columns = list(embedding_record.keys())
                            placeholders = ["%s"] * len(columns)
                            
                            sql = f"""
                            INSERT INTO song_embeddings_vector 
                            ({', '.join(columns)})
                            VALUES ({', '.join(placeholders)})
                            """
                            
                            values = tuple(embedding_record.values())
                            try:
                                cursor.execute(sql, values)
                            except Exception as e:
                                logger.error(f"Error inserting embedding for {song['song_id']}: {str(e)}")
                    
                    # Log column stats for this batch
                    logger.info(f"Embedding column coverage in batch {i+1}:")
                    for col, count in embedding_column_counts.items():
                        logger.info(f"Column '{col}' present in {count}/{len(batch)} embeddings ({count/len(batch)*100:.2f}%)")
                    
                    # Commit after each batch
                    conn.commit()
                    logger.info(f"Batch {i+1} committed")
                
                logger.info(f"Successfully loaded {len(new_songs)} new records")
            else:
                logger.error("Songs table doesn't exist in the database. Please create it first.")
                raise ValueError("Songs table doesn't exist")
        
        except Exception as e:
            logger.error(f"Error during data loading: {str(e)}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()

    logger.info("Database operation completed successfully.")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise