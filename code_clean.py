import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load dataset
df = pd.read_csv("Data/Raw/SpotifyFeatures.csv")
print(f"Initial size: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Handle missing values
df.dropna(inplace=True)

# 3. Remove duplicate rows
df.drop_duplicates(inplace=True)

df['artist'] = df['artist_name']

# 5. Encode categorical columns
genre_encoder = LabelEncoder()
artist_encoder = LabelEncoder()
df['genre'] = genre_encoder.fit_transform(df['genre'])
df['artist_name'] = artist_encoder.fit_transform(df['artist_name'])

# 6. Normalize numerical features
scaler = StandardScaler()
num_features = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'popularity']
df[num_features] = scaler.fit_transform(df[num_features])

# 7. Save cleaned dataset
df.to_csv("Data/Processed/spotify_clean.csv", index=False)
print("✅ Cleaned dataset saved as: spotify_clean.csv")

# 8. Display summary statistics
print(df.describe().T)
