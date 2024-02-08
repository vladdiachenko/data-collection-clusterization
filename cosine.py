from transformers import BertModel, BertTokenizer
import torch
from scipy.spatial.distance import cosine

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

tokens_cat = tokenizer("Retrospective", return_tensors="pt")
tokens_dog = tokenizer("Retrospektiven", return_tensors="pt")

# Get BERT embeddings
with torch.no_grad():
    embeddings_cat = model(**tokens_cat).last_hidden_state.mean(dim=1).squeeze().numpy()
    embeddings_dog = model(**tokens_dog).last_hidden_state.mean(dim=1).squeeze().numpy()

# Calculate cosine similarity
cosine_similarity = 1 - cosine(embeddings_cat, embeddings_dog)

print(f"Cosine Similarity: {cosine_similarity}")
