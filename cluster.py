import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')  # Download the punkt tokenizer

# Load your CSV file with reduced embeddings in scientific notation
file_path = 'reduced.csv'
df = pd.read_csv(file_path)

# Assuming your embeddings are in a column named 'embeddings'
embeddings_column = 'embeddings'

# Convert the string representations of embeddings to actual lists
df[embeddings_column] = df[embeddings_column].apply(lambda x: [float(val) for val in x.replace('E+', 'e').replace('E-', 'e-').split(';')])

# Stack the embeddings into a 2D array
embedding_array = np.vstack(df[embeddings_column])

# Standardize the data (important for KMeans)
scaler = StandardScaler()
embedding_array_standardized = scaler.fit_transform(embedding_array)

num_clusters = 11
kmeans = KMeans(n_clusters=num_clusters)
df['cluster'] = kmeans.fit_predict(embedding_array)

df.to_csv('clustered_results.csv', index=False)