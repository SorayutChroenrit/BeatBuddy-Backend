import pandas as pd
import re
import json
import uuid
import ollama  # Make sure you have the ollama package installed

# 1. Load data from CSV file
print("Loading song data...")
file_path = "data/raw/raw_english_songs_lyrics.csv" 
print(f"Attempting to load file from: {file_path}")
try:
    df = pd.read_csv(file_path)
    print("File loaded successfully!")
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# 2. Check basic information about the dataset
print(f"Dataset shape: {df.shape}")
print(f"Columns found: {df.columns.tolist()}")
print(f"Missing values:\n{df.isnull().sum()}")

# 3. Clean lyrics while preserving original
def clean_lyrics(text):
    if pd.isna(text):
        return {
            'original_lyrics': '',
            'cleaned_lyrics': ''
        }
    
    # Keep original lyrics exactly as they are
    original_lyrics = str(text)
    
    # Remove punctuation and convert to lowercase, but keep numbers
    cleaned_text = re.sub(r'[^\w\s]', ' ', original_lyrics.lower())
    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return {
        'original_lyrics': original_lyrics, 
        'cleaned_lyrics': cleaned_text
    }

# Process lyrics - use the correct column name 'Lyrics' as shown in CSV
print("Cleaning lyrics while preserving originals...")
lyrics_processed = df['Lyrics'].apply(clean_lyrics)
df['original_lyrics'] = lyrics_processed.apply(lambda x: x['original_lyrics'])
df['cleaned_lyrics'] = lyrics_processed.apply(lambda x: x['cleaned_lyrics'])

# 4. Create metadata field
print("Creating metadata field...")
df['metadata'] = "track: " + df['Song Title'].str.lower() + " artist: " + df['Artist'].str.lower()

# 5. Create Embeddings with Ollama
print("Generating embeddings with Ollama (Llama 3.1 model)...")

# Function to get embeddings from Ollama
def get_ollama_embedding(text):
    try:
        response = ollama.embeddings(model='llama3.1', prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Return zero vector in case of error
        return [0] * 3072  # Adjust size according to the output dimension of llama3.1

# Process in batches to save memory
batch_size = 32

# Generate embeddings for lyrics
print("Generating lyrics embeddings...")
lyrics_embeddings = []
for i in range(0, len(df), batch_size):
    print(f"Processing lyrics batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
    batch = df['cleaned_lyrics'][i:i+batch_size].tolist()
    for text in batch:
        # Truncate text for efficiency
        truncated_text = text[:1000]  # Limit to 1000 characters
        embedding = get_ollama_embedding(truncated_text)
        lyrics_embeddings.append(embedding)

# Generate embeddings for metadata
print("Generating metadata embeddings...")
metadata_embeddings = []
for i in range(0, len(df), batch_size):
    print(f"Processing metadata batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
    batch = df['metadata'][i:i+batch_size].tolist()
    for text in batch:
        embedding = get_ollama_embedding(text)
        metadata_embeddings.append(embedding)

# Add embeddings to dataframe
df['lyrics_embedding'] = lyrics_embeddings
df['metadata_embedding'] = metadata_embeddings

# 6. Handle IDs
# Store original song_id as siamzone_id
if 'Song ID' in df.columns:
    df['siamzone_id'] = df['Song ID']  # Keep original format
    print("Successfully mapped original song IDs to siamzone_id field")
else:
    df['siamzone_id'] = None
    print("WARNING: No Song ID column found in original dataset")

# Create new UUIDs for primary key
df['song_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

# 7. Prepare data for output JSON and CSV
print("Preparing final data records...")
song_records = []

for idx, row in df.iterrows():
    # Combine fields according to CSV structure
    song_record = {
        'song_id': row['song_id'],
        'track_name': row['Song Title'],
        'track_artist': row['Artist'],
        'original_lyrics': row['original_lyrics'],
        'lyrics': row['cleaned_lyrics'],
        'lyrics_embedding': row['lyrics_embedding'],
        'metadata_embedding': row['metadata_embedding'],
        'siamzone_id': row['siamzone_id']  # Original song_id
    }
    song_records.append(song_record)

# 8. Save processed data as JSON
print("Saving to JSON...")
with open('english_processed_songs.json', 'w', encoding='utf-8') as f:
    json.dump(song_records, f, ensure_ascii=False)

# 9. Save data as CSV
print("Saving combined data to CSV...")
df_for_csv = df.copy()
# Convert embedding lists to JSON strings for CSV storage
df_for_csv['lyrics_embedding'] = df_for_csv['lyrics_embedding'].apply(json.dumps)
df_for_csv['metadata_embedding'] = df_for_csv['metadata_embedding'].apply(json.dumps)

# Rename columns to match the desired output format
df_for_csv = df_for_csv.rename(columns={
    'Song Title': 'track_name',
    'Artist': 'track_artist'
})

# Save as combined CSV
df_for_csv.to_csv('english_processed_songs.csv', index=False, encoding='utf-8')

print("Data preprocessing completed and saved to files:")
print("- english_processed_songs.json: Complete JSON data")
print("- english_processed_songs.csv: Combined data in CSV format")