import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your CSV file with reduced embeddings in scientific notation
file_path = 'reduced.csv'
df = pd.read_csv(file_path)

# Assuming your embeddings are in a column named 'embeddings'
embeddings_column = 'embeddings'

# Convert semicolon-separated string embeddings to numerical values for the "embeddings" column
def parse_embedding(embedding_str):
    # Split the string into individual components
    components = embedding_str.split(';')
    
    # Convert each component to float
    return [float(component) for component in components]

# Apply parsing to the "embeddings" column
df[embeddings_column] = df[embeddings_column].apply(parse_embedding)

# Stack the embeddings into a 2D array
embedding_array = np.vstack(df[embeddings_column])

# Standardize the data (important for KMeans)
scaler = StandardScaler()
embedding_array_standardized = scaler.fit_transform(embedding_array)

# Perform clustering using K-Means
def perform_clustering(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels

# Calculate silhouette score
def calculate_silhouette_score(embeddings, labels):
    silhouette_avg = silhouette_score(embeddings, labels)
    return silhouette_avg

max = 0
maxNum = 0

for k in range(2, 4):
     # Perform clustering
    labels = perform_clustering(embedding_array_standardized, k)
    # Calculate silhouette score
    silhouette_avg = calculate_silhouette_score(embedding_array_standardized, labels)
    if (silhouette_avg > max):
        max = silhouette_avg
        maxNum = k

print(f"Silhouette Score for {k} clusters: {max}")
