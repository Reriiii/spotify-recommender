import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load dataset
df = pd.read_csv("SpotifyFeatures.csv")
print(f"Initial size: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Handle missing values
df.dropna(inplace=True)
print(f"After removing NaN: {df.shape[0]} rows remain")

# 3. Remove duplicate rows
df.drop_duplicates(inplace=True)
print(f"After removing duplicates: {df.shape[0]} rows remain")

# 4. Select relevant features
features = [
    'track_name', 'artist_name', 'genre',
    'danceability', 'energy', 'valence',
    'tempo', 'loudness', 'popularity'
]
df = df[features]

# 5. Encode categorical columns
encoder = LabelEncoder()
df['genre'] = encoder.fit_transform(df['genre'])
df['artist_name'] = encoder.fit_transform(df['artist_name'])

# 6. Normalize numerical features
scaler = StandardScaler()
num_features = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'popularity']
df[num_features] = scaler.fit_transform(df[num_features])

# 7. Save cleaned dataset
df.to_csv("spotify_clean.csv", index=False)
print("Cleaned dataset saved as: spotify_clean.csv")

# 8. Display summary statistics
print(df.describe().T)
