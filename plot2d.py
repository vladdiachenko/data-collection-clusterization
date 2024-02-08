#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot(input_filename, embeddings_column, label_column):
    df = pd.read_csv(input_filename)

    df[embeddings_column] = df[embeddings_column].apply(lambda x: [float(i) for i in x.split(';')])

    df['X'] = df[embeddings_column].apply(lambda x: x[0])
    df['Y'] = df[embeddings_column].apply(lambda x: x[1])

    plt.figure(figsize=(10, 6))
    for _, row in df.iterrows():
        plt.scatter(row['X'], row['Y'], marker='o', color='blue')
        plt.text(row['X'], row['Y'], row[label_column], fontsize=9)
    
    plt.title('Embeddings')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

plot('reduced.csv', 'embeddings', 'OTHER_USE_CASE')