import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("Data/Raw/SpotifyFeatures.csv")
print(f"Initial size: {df.shape[0]} rows, {df.shape[1]} columns")

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df['artist'] = df['artist_name']

for col in ['genre', 'artist_name']:
    df[col] = LabelEncoder().fit_transform(df[col])

scaler = StandardScaler()
num_features = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'popularity']
df[num_features] = scaler.fit_transform(df[num_features])

df.to_csv("Data/Processed/spotify_clean.csv", index=False)
print("✅ Cleaned dataset saved as: spotify_clean.csv")

# 8. Display summary statistics
print(df.describe().T)
