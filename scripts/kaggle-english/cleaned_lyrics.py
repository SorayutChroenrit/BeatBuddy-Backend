import pandas as pd
import re
import json
import uuid
import ollama

# 1. โหลดข้อมูลจากไฟล์ CSV
print("Loading song data...")
file_path = "data/raw/cleaned_lyrics_with_artist_names.csv"  
print(f"Attempting to load file from: {file_path}")
df = pd.read_csv(file_path)

# 2. ตรวจสอบข้อมูลพื้นฐานของ dataset
print(f"Dataset shape: {df.shape}")
print(f"Columns found: {df.columns.tolist()}")
print(f"Missing values:\n{df.isnull().sum()}")

# 3. ทำความสะอาดเนื้อเพลง
def clean_lyrics(text):
    if pd.isna(text):
        return {
            'original_lyrics': '',
            'cleaned_lyrics': ''
        }
    
    # เก็บเนื้อเพลงดั้งเดิม
    original_lyrics = str(text)
    
    # ลบเครื่องหมายวรรคตอนและแปลงเป็นตัวพิมพ์เล็ก แต่เก็บตัวเลขไว้
    cleaned_text = re.sub(r'[^\w\s]', ' ', original_lyrics.lower())
    # ลบช่องว่างที่ซ้ำกัน
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return {
        'original_lyrics': original_lyrics,
        'cleaned_lyrics': cleaned_text
    }

# ประมวลผลเนื้อเพลง - ใช้ชื่อคอลัมน์ที่ถูกต้อง 'lyrics' ตามที่แสดงใน CSV
print("Cleaning lyrics...")
lyrics_processed = df['lyrics'].apply(clean_lyrics)
df['original_lyrics'] = lyrics_processed.apply(lambda x: x['original_lyrics'])
df['cleaned_lyrics'] = lyrics_processed.apply(lambda x: x['cleaned_lyrics'])

# 4. สร้างฟิลด์ metadata
print("Creating metadata field...")
df['metadata'] = "track: " + df['song'].str.lower() + " artist: " + df['artists'].str.lower()

# 5. สร้าง Embeddings ด้วย Ollama
print("Generating embeddings with Ollama (Llama 3.1 model)...")

# ฟังก์ชันเพื่อรับ embeddings จาก Ollama
def get_ollama_embedding(text):
    try:
        response = ollama.embeddings(model='llama3.1', prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # ส่งคืนเวกเตอร์ศูนย์ในกรณีที่เกิดข้อผิดพลาด
        return [0] * 3072  # ปรับขนาดตาม output dimension ของ llama3.1

# ประมวลผลเป็นกลุ่มเพื่อประหยัดหน่วยความจำ
batch_size = 32

# สร้าง embeddings สำหรับเนื้อเพลง
print("Generating lyrics embeddings...")
lyrics_embeddings = []
for i in range(0, len(df), batch_size):
    print(f"Processing lyrics batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
    batch = df['cleaned_lyrics'][i:i+batch_size].tolist()
    for text in batch:
        # ตัดข้อความให้สั้นลงเพื่อประสิทธิภาพ
        truncated_text = text[:1000]  # จำกัดที่ 1000 ตัวอักษร
        embedding = get_ollama_embedding(truncated_text)
        lyrics_embeddings.append(embedding)

# สร้าง embeddings สำหรับ metadata
print("Generating metadata embeddings...")
metadata_embeddings = []
for i in range(0, len(df), batch_size):
    print(f"Processing metadata batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
    batch = df['metadata'][i:i+batch_size].tolist()
    for text in batch:
        embedding = get_ollama_embedding(text)
        metadata_embeddings.append(embedding)

# เพิ่ม embeddings ลงใน dataframe
df['lyrics_embedding'] = lyrics_embeddings
df['metadata_embedding'] = metadata_embeddings

# 6. จัดการ IDs
# เก็บ song_id ดั้งเดิมเป็น spotify_id และเก็บ artist_id ไว้
df['spotify_id'] = df['song_id']  # song_id ดั้งเดิมคือ Spotify ID
df['original_artist_id'] = df['artist_id']

# สร้าง UUIDs ใหม่สำหรับคีย์หลัก
df['new_song_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

# 7. เตรียมข้อมูลสำหรับ output JSON และ CSV
print("Preparing final data records...")
song_records = []

for idx, row in df.iterrows():
    # รวมฟิลด์ตามโครงสร้าง CSV
    song_record = {
        'song_id': row['new_song_id'],
        'track_name': row['song'],
        'track_artist': row['artists'],
        'original_lyrics': row['original_lyrics'],
        'lyrics': row['cleaned_lyrics'],
        'lyrics_embedding': row['lyrics_embedding'],
        'metadata_embedding': row['metadata_embedding'],
        'spotify_id': row['spotify_id'],
        'original_artist_id': row['original_artist_id'],
        'genres': row['genres'],
        'explicit': row['explicit']
    }
    song_records.append(song_record)

# 8. บันทึกข้อมูลที่ประมวลผลแล้วเป็น JSON ตามที่ต้องการ
print("Saving to JSON...")
with open('kaggle_processed_songs.json', 'w') as f:
    json.dump(song_records, f)

# 9. บันทึกข้อมูลเป็นไฟล์ CSV รวม
print("Saving combined data to CSV...")
df_for_csv = df.copy()
df_for_csv['lyrics_embedding'] = df_for_csv['lyrics_embedding'].apply(json.dumps)
df_for_csv['metadata_embedding'] = df_for_csv['metadata_embedding'].apply(json.dumps)

# บันทึกเป็น CSV รวม
df_for_csv.to_csv('kaggle_processed_songs.csv', index=False)

print("Data preprocessing completed and saved to files:")
print("- kaggle_processed_songs.json: Complete JSON data")
print("- kaggle_processed_songs.csv: Combined data in CSV format")