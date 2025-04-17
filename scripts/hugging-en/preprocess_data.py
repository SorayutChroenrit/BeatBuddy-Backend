import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import re
import json
import uuid
import ollama  

# 1. โหลดข้อมูล
df = pd.read_csv("hf://datasets/Zuru7/Spotify_Songs_with_SoundCloud_links/song_df_normalised.csv")

# 2. ทำความสะอาดข้อมูล
# 2.1 ตรวจสอบและจัดการค่า null
print(f"Missing values before cleaning:\n{df.isnull().sum()}")

# 2.2 ตรวจสอบค่าซ้ำ
duplicate_rows = df.duplicated().sum()
print(f"Duplicate rows: {duplicate_rows}")
if duplicate_rows > 0:
    df = df.drop_duplicates()

# 2.3 ทำความสะอาดข้อมูลเนื้อเพลง
def clean_lyrics(text):
    if pd.isna(text):
        return {
            'original_lyrics': '',
            'cleaned_lyrics': ''
        }
    
    # เก็บเนื้อเพลงต้นฉบับ
    original_lyrics = str(text)
    
    # ลบเครื่องหมายวรรคตอนและแปลงเป็นตัวพิมพ์เล็ก แต่เก็บตัวเลขไว้
    cleaned_text = re.sub(r'[^\w\s]', ' ', original_lyrics.lower())
    # ลบช่องว่างที่ซ้ำกัน
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return {
        'original_lyrics': original_lyrics,
        'cleaned_lyrics': cleaned_text
    }

# แยกคอลัมน์สำหรับเนื้อเพลง
lyrics_processed = df['lyrics'].apply(clean_lyrics)
df['original_lyrics'] = lyrics_processed.apply(lambda x: x['original_lyrics'])
df['cleaned_lyrics'] = lyrics_processed.apply(lambda x: x['cleaned_lyrics'])

# 2.4 ทำความสะอาดชื่อเพลงและศิลปิน
df['cleaned_track_name'] = df['track_name'].str.lower().str.strip()
df['cleaned_artist'] = df['track_artist'].str.lower().str.strip()

# 3. Feature Engineering
# 3.1 สร้าง metadata ที่รวมข้อมูลสำคัญ
df['metadata'] = (
    "track: " + df['cleaned_track_name'] + 
    " artist: " + df['cleaned_artist'] + 
    " album: " + df['track_album_name'].str.lower() + 
    " genre: " + df['playlist_genre'].str.lower() + 
    " subgenre: " + df['playlist_subgenre'].str.lower()
)

# 3.2 ดึงคุณลักษณะทางดนตรี (เก็บเป็น feature vector)
audio_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo'
]

# สเกลข้อมูลคุณลักษณะเสียงให้อยู่ในช่วงที่เหมาะสม
scaler = StandardScaler()
df[audio_features] = scaler.fit_transform(df[audio_features])

# 3.3 แปลงข้อมูล categorical เป็น one-hot encoding
categorical_features = ['playlist_genre', 'playlist_subgenre', 'language', 'sentiment']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(
    encoded_cats, 
    columns=encoder.get_feature_names_out(categorical_features)
)

# เชื่อมต่อข้อมูลที่ encoding แล้วกับ dataframe หลัก
df = pd.concat([df, encoded_df], axis=1)

# 4. สร้าง Embeddings ด้วย Ollama และ Llama 3.1
print("Generating embeddings with Ollama (Llama 3.1 model)...")

# 4.1 ฟังก์ชันสำหรับสร้าง embeddings ด้วย Ollama
def get_ollama_embedding(text):
    try:
        response = ollama.embeddings(model='llama3.1', prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # ส่งคืน vector ที่มีค่าเป็น 0 ในกรณีเกิดข้อผิดพลาด (ขึ้นอยู่กับขนาด embeddings ของ llama3.1)
        return [0] * 3072  # ปรับขนาดตามที่ llama3.1 ให้มา

# 4.2 สร้าง embeddings ทีละส่วน (batch processing) เพื่อลดการใช้หน่วยความจำ
batch_size = 32  # ขนาด batch ที่เล็กลงเพราะ LLM ต้องการทรัพยากรมากกว่า

# 4.3 สร้าง embeddings สำหรับเนื้อเพลง
lyrics_embeddings = []
for i in range(0, len(df), batch_size):
    print(f"Processing lyrics batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
    batch = df['cleaned_lyrics'][i:i+batch_size].tolist()
    for text in batch:
        # ตัดข้อความให้สั้นลงเพื่อประสิทธิภาพ (Llama อาจมีข้อจำกัดเรื่องความยาว)
        truncated_text = text[:1000]  # ตัดข้อความให้ไม่เกิน 1000 ตัวอักษร
        embedding = get_ollama_embedding(truncated_text)
        lyrics_embeddings.append(embedding)

# 4.4 สร้าง embeddings สำหรับ metadata
metadata_embeddings = []
for i in range(0, len(df), batch_size):
    print(f"Processing metadata batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
    batch = df['metadata'][i:i+batch_size].tolist()
    for text in batch:
        embedding = get_ollama_embedding(text)
        metadata_embeddings.append(embedding)

# 4.5 เก็บ embeddings ในแต่ละแถวของ DataFrame
df['lyrics_embedding'] = lyrics_embeddings
df['metadata_embedding'] = metadata_embeddings

# 5. คำนวณคะแนนความเป็นที่นิยม (Normalized)
df['popularity_score'] = df['track_popularity'] / 100.0

# 6. สร้าง song_id ด้วย UUID และเก็บ original IDs
# สร้าง UUID สำหรับแต่ละเพลง
df['song_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

# เพิ่มคอลัมน์สำหรับเก็บ ID จากแหล่งข้อมูลต่างๆ
# ในกรณีที่มี spotify_id อยู่แล้วในข้อมูล (ตรวจสอบว่ามีคอลัมน์นี้หรือไม่)
if 'spotify_id' not in df.columns:
    # ถ้าไม่มี spotify_id ในข้อมูลแต่เราสามารถดึงได้จากบางแหล่ง (เช่น จาก links หรือข้อมูลอื่น)
    # สมมติว่าเราสามารถแยก ID จาก URL ได้ (ปรับตามรูปแบบของข้อมูลจริง)
    spotify_pattern = r'spotify\.com/track/([a-zA-Z0-9]+)'
    df['spotify_id'] = df['links'].str.extract(spotify_pattern, expand=False)

# ในกรณีที่มี siamzone_id (ถ้าไม่มีให้ตั้งเป็น None)
df['siamzone_id'] = None  # ตั้งค่าเริ่มต้นเป็น None

# 7. เตรียมข้อมูลสำหรับเก็บในไฟล์ CSV และ JSON
song_records = []

for idx, row in df.iterrows():
    # สร้าง dictionary ที่มีค่า audio features แต่ละค่าแยกเป็นฟิลด์
    song_record = {
        'song_id': row['song_id'],
        'track_name': row['track_name'],
        'track_artist': row['track_artist'],
        'track_album_name': row['track_album_name'],
        'original_lyrics': row['original_lyrics'],
        'lyrics': row['cleaned_lyrics'], 
        'lyrics_embedding': row['lyrics_embedding'],  
        'metadata_embedding': row['metadata_embedding'],
        'playlist_genre': row['playlist_genre'],
        'playlist_subgenre': row['playlist_subgenre'],
        'language': row['language'],
        'sentiment': row['sentiment'],
        'popularity_score': float(row['popularity_score']),
        'link': row['links'],
        'spotify_id': row['spotify_id'],
        'siamzone_id': row['siamzone_id'],
        # เพิ่ม audio features แต่ละตัวแยกเป็นฟิลด์
        'danceability': float(row['danceability']),
        'energy': float(row['energy']),
        'key': float(row['key']),
        'loudness': float(row['loudness']),
        'mode': float(row['mode']),
        'speechiness': float(row['speechiness']),
        'acousticness': float(row['acousticness']),
        'instrumentalness': float(row['instrumentalness']),
        'liveness': float(row['liveness']),
        'valence': float(row['valence']),
        'tempo': float(row['tempo'])
    }
    song_records.append(song_record)

# 8. บันทึกข้อมูลที่ประมวลผลแล้วเป็นไฟล์ JSON ตามชื่อที่ต้องการ
with open('huggingface_processed_songs.json', 'w') as f:
    json.dump(song_records, f)

# 9. บันทึกเป็นไฟล์ CSV โดยเก็บแต่ละ audio feature เป็นคอลัมน์แยกกัน
df_for_csv = df.copy()
df_for_csv['lyrics_embedding'] = df_for_csv['lyrics_embedding'].apply(json.dumps)
df_for_csv['metadata_embedding'] = df_for_csv['metadata_embedding'].apply(json.dumps)

# ไม่จำเป็นต้องสร้าง audio_features_json อีกต่อไป เพราะเราเก็บแต่ละ feature เป็นคอลัมน์แยกแล้ว
# df_for_csv['audio_features_json'] = df_for_csv[audio_features].apply(lambda row: json.dumps(row.tolist()), axis=1)

# บันทึกเป็นไฟล์ CSV รวมตามชื่อที่ต้องการ (จะมีคอลัมน์แยกสำหรับแต่ละ audio feature)
df_for_csv.to_csv('huggingface_processed_songs.csv', index=False)

print("Data preprocessing with Llama 3.1 completed.")
print("Files saved: huggingface_processed_songs.json, huggingface_processed_songs.csv")