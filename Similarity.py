import pandas as pd
import numpy as np

# Đọc dữ liệu
df = pd.read_csv('spotify_clean.csv')

# Các cột đặc trưng
feature_cols = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'popularity']
df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)

# Ma trận đặc trưng
X = df_clean[feature_cols].values.astype(float)

# Chuẩn hóa
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_norm = (X - mean) / std

# Hàm tính cosine similarity giữa 1 bài hát và tất cả bài khác
def recommend_similar(song_index, top_k=5):
    target = X_norm[song_index]
    norms = np.linalg.norm(X_norm, axis=1)
    dot_products = np.dot(X_norm, target)
    target_norm = np.linalg.norm(target)
    similarities = dot_products / (norms * target_norm)

    # Lấy top-k bài giống nhất (trừ chính nó)
    indices = np.argsort(-similarities)
    indices = [i for i in indices if i != song_index][:top_k]
    return df_clean.iloc[indices][['track_name', 'artist_name', 'genre']]

# Ví dụ: gợi ý cho bài hát ở dòng số 1000
print(recommend_similar(1000, top_k=5))
