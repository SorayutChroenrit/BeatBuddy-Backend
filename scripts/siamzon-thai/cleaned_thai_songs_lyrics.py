import pandas as pd
import re
import json
import uuid
import ollama
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords

# 1. โหลดข้อมูลจากไฟล์ CSV
print("Loading Thai song data...")
file_path = "data/cleaned_thai_songs_lyrics.csv"  
print(f"Attempting to load file from: {file_path}")
df = pd.read_csv(file_path)

# 2. ตรวจสอบข้อมูลพื้นฐานของ dataset
print(f"Dataset shape: {df.shape}")
print(f"Columns found: {df.columns.tolist()}")
print(f"Missing values:\n{df.isnull().sum()}")

# 3. ทำความสะอาดและประมวลผลเนื้อเพลงภาษาไทย
def process_thai_lyrics(text):
    if pd.isna(text):
        return {
            'original_lyrics': '',
            'cleaned_lyrics': '',
            'tokenized_lyrics': '',
            'filtered_lyrics': ''
        }
    
    # เก็บเนื้อเพลงดั้งเดิม
    original_lyrics = str(text)
    
    # ลบเครื่องหมายวรรคตอนและช่องว่างที่ซ้ำกัน แต่เก็บตัวเลขไว้
    cleaned_text = re.sub(r'[^\u0E00-\u0E7F\w\s]', ' ', original_lyrics)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # ทำ word tokenization ด้วย PyThaiNLP
    tokenized_text = word_tokenize(cleaned_text, engine='newmm')
    tokenized_string = ' '.join(tokenized_text)
    
    # กรองคำหยุด (stopwords)
    thai_stopwords_list = list(thai_stopwords())
    filtered_tokens = [word for word in tokenized_text if word not in thai_stopwords_list and len(word) > 1]
    filtered_string = ' '.join(filtered_tokens)
    
    return {
        'original_lyrics': original_lyrics,
        'cleaned_lyrics': cleaned_text,
        'tokenized_lyrics': tokenized_string,
        'filtered_lyrics': filtered_string
    }

# ประมวลผลเนื้อเพลง - ใช้ชื่อคอลัมน์ที่ถูกต้องตามที่อยู่ใน CSV
print("Processing Thai lyrics...")
lyrics_column = 'lyrics' if 'lyrics' in df.columns else 'Lyrics'
lyrics_processed = df[lyrics_column].apply(process_thai_lyrics)

df['original_lyrics'] = lyrics_processed.apply(lambda x: x['original_lyrics'])
df['cleaned_lyrics'] = lyrics_processed.apply(lambda x: x['cleaned_lyrics'])
df['tokenized_lyrics'] = lyrics_processed.apply(lambda x: x['tokenized_lyrics'])
df['filtered_lyrics'] = lyrics_processed.apply(lambda x: x['filtered_lyrics'])

# 4. สร้างฟิลด์ metadata
print("Creating metadata field...")
song_column = 'song' if 'song' in df.columns else 'Song Title'
artist_column = 'artists' if 'artists' in df.columns else 'Artist'

df['metadata'] = "track: " + df[song_column].str.lower() + " artist: " + df[artist_column].str.lower()

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

# สร้าง embeddings สำหรับเนื้อเพลงที่ผ่านการ tokenize และกรองแล้ว
print("Generating lyrics embeddings...")
lyrics_embeddings = []
for i in range(0, len(df), batch_size):
    print(f"Processing lyrics batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
    batch = df['filtered_lyrics'][i:i+batch_size].tolist()
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
# เก็บ song_id ดั้งเดิมเป็น siamzone_id สำหรับเพลงไทย
id_column = 'song_id' if 'song_id' in df.columns else 'Song ID'
if id_column in df.columns:
    df['siamzone_id'] = df[id_column]
    print(f"Successfully mapped original {id_column} to siamzone_id field")
else:
    df['siamzone_id'] = None
    print("WARNING: No song ID column found in original dataset")

# สร้าง UUIDs ใหม่สำหรับคีย์หลัก
df['song_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

# 7. เตรียมข้อมูลสำหรับ output JSON และ CSV
print("Preparing final data records...")
song_records = []

for idx, row in df.iterrows():
    # รวมฟิลด์ตามโครงสร้าง
    song_record = {
        'song_id': row['song_id'],
        'track_name': row[song_column],
        'track_artist': row[artist_column],
        'original_lyrics': row['original_lyrics'],
        'cleaned_lyrics': row['cleaned_lyrics'],
        'tokenized_lyrics': row['tokenized_lyrics'],
        'filtered_lyrics': row['filtered_lyrics'],
        'lyrics_embedding': row['lyrics_embedding'],
        'metadata_embedding': row['metadata_embedding'],
        'siamzone_id': row['siamzone_id'],
        'language': 'thai'
    }
    song_records.append(song_record)

# 8. บันทึกข้อมูลที่ประมวลผลแล้วเป็น JSON
print("Saving to JSON...")
with open('thai_processed_songs.json', 'w', encoding='utf-8') as f:
    json.dump(song_records, f, ensure_ascii=False)

# 9. บันทึกข้อมูลเป็นไฟล์ CSV รวม
print("Saving combined data to CSV...")
df_for_csv = df.copy()
df_for_csv['lyrics_embedding'] = df_for_csv['lyrics_embedding'].apply(json.dumps)
df_for_csv['metadata_embedding'] = df_for_csv['metadata_embedding'].apply(json.dumps)

# บันทึกเป็น CSV รวม
df_for_csv.to_csv('thai_processed_songs.csv', index=False, encoding='utf-8')

print("Data preprocessing completed and saved to files:")
print("- thai_processed_songs.json: Complete JSON data with Thai-specific processing")
print("- thai_processed_songs.csv: Combined data in CSV format")