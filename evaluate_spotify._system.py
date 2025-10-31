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
import time

# Đánh giá thời gian chạy trung bình
def evaluate_performance(n_tests=100, top_k=5):
    indices = np.random.choice(len(df_clean), n_tests, replace=False)
    start_time = time.time()
    for i in indices:
        recommend_similar(i, top_k=top_k)
    end_time = time.time()
    avg_time = (end_time - start_time) / n_tests
    print(f"Trung bình mỗi lần gợi ý mất: {avg_time:.4f} giây")

# Gọi hàm đánh giá
evaluate_performance(n_tests=100, top_k=5)
def evaluate_similarity_quality(n_tests=100, top_k=5):
    indices = np.random.choice(len(df_clean), n_tests, replace=False)
    total_avg_sim = 0
    for i in indices:
        target = X_norm[i]
        norms = np.linalg.norm(X_norm, axis=1)
        dot_products = np.dot(X_norm, target)
        target_norm = np.linalg.norm(target)
        similarities = dot_products / (norms * target_norm)
        similarities[i] = -1  # loại bỏ chính nó
        top_sims = np.sort(similarities)[-top_k:]
        total_avg_sim += np.mean(top_sims)
    print(f"Trung bình độ tương đồng cosine top-{top_k}: {total_avg_sim / n_tests:.4f}")


# Gọi hàm đánh giá
evaluate_similarity_quality(n_tests=100, top_k=5)



