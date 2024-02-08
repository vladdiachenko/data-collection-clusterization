#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import csv

def process(input_file, dimensions, output_file):
    # Load the dataset
    df = pd.read_csv(input_file)

    embeddings_column = 'embeddings'

    # Convert the string representations of embeddings to actual lists
    df[embeddings_column] = df[embeddings_column].apply(lambda x: [float(val) for val in x.replace('E+', 'e').replace('E-', 'e-').split(';')])

    # Stack the embeddings into a 2D array
    embedding_array = np.vstack(df[embeddings_column])

    # Standardize the data (important for t-SNE)
    scaler = StandardScaler()
    embedding_array_standardized = scaler.fit_transform(embedding_array)

    # Perform t-SNE
    tsne = TSNE(n_components=dimensions, metric='cosine', random_state=42)
    reduced_embeddings = tsne.fit_transform(embedding_array_standardized)
    df['embeddings'] = pd.DataFrame(reduced_embeddings).apply(lambda row: ';'.join([f'{x:.9E}' for x in row]), axis=1)
    df.to_csv(output_file, index=False)

# Example usage
pd.set_option('display.max_rows', None)

process('embeddings.csv', 2, 'reduced.csv')
