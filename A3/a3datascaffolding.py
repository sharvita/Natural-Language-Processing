'''
CSCI 5832 Assignment 3
Spring 2025
Use this code as a starting point or reference for the data preprocessing portion 
of your assignment.
'''

import pandas as pd
# Load the Rotten Tomatoes polarity dataset
def load_rt_dataset():
    reviews = []
    for sentiment in ['pos', 'neg']:
        path = f'rt-polarity.{sentiment}'
        file = open(path)
        for line in file.readlines():
            review = line.strip()
            reviews.append({'review': review, 'sentiment': sentiment})
    return pd.DataFrame(reviews)

reviews = load_rt_dataset()
print(reviews.head())

# Tokenization and cleaning
# suggested packages... hint hint...
import re
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('punkt_tab')

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess(text : str) -> list:
    # your code here
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def getConversionDict():
    reviews = load_rt_dataset()
    # Preprocess all documents
    preprocessed_documents = reviews['review'].apply(preprocess)
    print(f'Number of preprocessed documents: {len(preprocessed_documents)}')
    print(preprocessed_documents.head())

    # Make a token2index dictionary and a index2token dictionary and convert the documents to sequences of indices
    token2index = {}
    index2token = {}
    index = 1 # reserve 0 for padding
    for document in preprocessed_documents:
        for token in document:
            # build the dictionaries, your code here
            if token not in token2index:
                token2index[token] = index
                index2token[index] = token
                index += 1
            pass

    token2index['[PAD]'] = 0
    index2token[0] = '[PAD]'

    print(f'Number of unique tokens: {len(token2index)}')
    return token2index, index2token


# Convert the dataset into sequences of indices
def document_to_sequence(document : str) -> list:
    return [token2index[token] for token in document]

preprocessed_documents = reviews['review'].apply(preprocess)
print(f'Number of preprocessed documents: {len(preprocessed_documents)}')
print(preprocessed_documents.head())
token2index = {}
index2token = {}
index = 1 # reserve 0 for padding
for document in preprocessed_documents:
    for token in document:
        # build the dictionaries, your code here
        if token not in token2index:
            token2index[token] = index
            index2token[index] = token
            index += 1
        pass

token2index['[PAD]'] = 0
index2token[0] = '[PAD]'

print(f'Number of unique tokens: {len(token2index)}')
sequences = preprocessed_documents.apply(document_to_sequence)
print(sequences.head()) # should now be a list of indices

# Truncate the sequences
def pad_sequence(sequence: list, max_length: int, padding_token: int = 0) -> list:
    # your code here
    if len(sequence) < max_length:
        return sequence + [padding_token] * (max_length - len(sequence))
    else:
        return sequence[:max_length]

# Maximum sequence length
max_length = 40

# Truncate the sequences
truncated_sequences = sequences.apply(lambda x: pad_sequence(x, max_length))

print(truncated_sequences.head())

# Finally, convert the sequences to tensors and create dataloaders 
# for training, validation, and testing
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

def getLoaders():

    reviews = load_rt_dataset()
    preprocessed_documents = reviews['review'].apply(preprocess)
    sequences = preprocessed_documents.apply(document_to_sequence)

    max_length = 40

    # Truncate the sequences
    truncated_sequences = sequences.apply(lambda x: pad_sequence(x, max_length))

    # Split the dataset into training and testing sets
    X_train, X_test_and_val, y_train, y_test_and_val = train_test_split(truncated_sequences, reviews['sentiment'], test_size=0.2, random_state=123)
    X_test, X_val, y_test, y_val = train_test_split(X_test_and_val, y_test_and_val, test_size=0.5, random_state=42)

    # Encode the target variable
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train.tolist())
    y_val = label_encoder.transform(y_val.tolist())
    y_test = label_encoder.transform(y_test.tolist())

    # Convert the vectorized reviews to numpy arrays
    X_train = torch.tensor(X_train.tolist())
    X_val = torch.tensor(X_val.tolist())
    X_test = torch.tensor(X_test.tolist())
    y_train = torch.tensor(y_train)
    y_val = torch.tensor(y_val)
    y_test = torch.tensor(y_test)

    # Define the dataset class
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    test_data = TensorDataset(X_test, y_test)

    # Define the dataloader
    batch_size = 16
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=len(test_data))


    # check the first sequence with index2token

    return train_loader, val_loader, test_loader
