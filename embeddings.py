#!/usr/bin/env python3
import os
import csv
import torch
from transformers import BertModel, BertTokenizer
import pandas as pd
from langdetect import detect

# Load your CSV file
file_path = 'data.csv'
df = pd.read_csv(file_path)

words_column = 'OTHER_USE_CASE'

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Set the model in evaluation mode
model.eval()

def filterData(text):
    if len(str(text)) <= 2:
        return False
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False

# Function to get embeddings for a sentence
def get_embedding(sentence):
    # Tokenize the sentence
    tokens = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)

    # Forward pass through the BERT model
    with torch.no_grad():
        outputs = model(**tokens)

    # Extract the embeddings for the [CLS] token (index 0)
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    # Convert to a list of strings in scientific notation format
    embedding_str = ';'.join(f'{val.item():.6e}' for val in cls_embedding[0])

    return embedding_str

# cleanup data
df[words_column] = df[words_column].apply(lambda sen: sen.strip("'\",."))
sanitized = df[words_column].apply(len)
df = df[df[words_column].apply(filterData)]
# Apply the function to obtain embeddings for each word in the column
df['embeddings'] = df[words_column].apply(get_embedding)

# Save the embeddings to a new CSV file
output_file_path = 'embeddings.csv'
df.to_csv(output_file_path, index=False)