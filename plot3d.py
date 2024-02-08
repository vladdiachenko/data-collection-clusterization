import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Load the CSV file with BERT embeddings
file_path = 'reduced.csv'
df = pd.read_csv(file_path)

# Assuming your embeddings are in a column named 'embeddings'
embeddings_column = 'embeddings'

# Convert the string representations of embeddings to actual lists
df[embeddings_column] = df[embeddings_column].apply(lambda x: [float(i) for i in x.split(';')])

# Stack the embeddings into a 2D array
embedding_array = np.vstack(df[embeddings_column])

# Standardize the data (important for t-SNE)
scaler = StandardScaler()
embedding_array_standardized = scaler.fit_transform(embedding_array)

# Choose a small perplexity value
perplexity_value = 10  # Start with a small value

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=3, perplexity=perplexity_value, random_state=42)
embedding_array_reduced = tsne.fit_transform(embedding_array_standardized)

# Create a new DataFrame with the reduced embeddings
columns = [f't-SNE_{i+1}' for i in range(3)]
df_reduced = pd.DataFrame(data=embedding_array_reduced, columns=columns)

# Concatenate the reduced embeddings with the original DataFrame
df_result = pd.concat([df, df_reduced], axis=1)

# Visualize in 3D using matplotlib
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(df_result['t-SNE_1'], df_result['t-SNE_2'], df_result['t-SNE_3'], s=10, alpha=0.5)

# Set labels
ax.set_xlabel('t-SNE_1')
ax.set_ylabel('t-SNE_2')
ax.set_zlabel('t-SNE_3')

# Set title
plt.title('t-SNE Visualization in 3D')

# Show the plot
plt.show()