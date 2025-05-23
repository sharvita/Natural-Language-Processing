from typing import List, Tuple
from sklearn.model_selection import train_test_split
import numpy as np
import csv

def load_train_data( positive_filepath: str, negative_filepath: str) -> Tuple[List[str], List[int]]:
    """Load the training data, producing Lists of text and labels
       
    Args:
        filepath (str): Path to the training file

    Returns:
        Tuple[List[str], List[int]]: The texts and labels
    """

    def _read(filename: str):
        texts = []
        with open(filename,"r") as f:
            for line in f:
                _id, text = line.rstrip().split("\t")
                texts.append(text)

        return texts

    texts = []
    labels = []
    for text in _read(positive_filepath):
        texts.append(text)
        labels.append(1)

    for text in _read(negative_filepath):
        texts.append(text)
        labels.append(0)

    return texts, labels


def load_test_data(filepath: str) -> List[str]:
    """Load the test data, producing a List of texts

    Args:
        filepath (str): Path to the training file

    Returns:
        List[str]: The texts
    """
    texts = []
    labels = []
    with open(filepath, "r") as file:
        for line in file:
            idx, text, label = line.rstrip().split("\t")
            texts.append(text)
            if label == 'POS':
                label = 1
            else:
                label = 0
            labels.append(label)

    return texts, labels

def split_data(all_texts, all_labels) :
    """Splits data the data into 80/20 split - 80 for training and 20 for test 

    Args:
        all texts and labels

    Returns:
        train_texts, train_labels, dev_texts, dev_labels
    """
    # all_texts = np.array(range(100))
    # all_labels = all_texts * 2

    # Split data into 80% training and 20% testing
    all_texts_train, all_texts_test, all_labels_train, all_labels_test = train_test_split(all_texts, all_labels, test_size=0.2, random_state=42)

    # Print the shapes of the resulting arrays
    # print("all_texts_train shape:", all_texts_train[2])
    # print("all_texts_test shape:", all_texts_test[2])
    # print("all_labels_train shape:", all_labels_train[3])
    # print("all_labels_test shape:", all_labels_test[2])

    return all_texts_train, all_labels_train, all_texts_test, all_labels_test

def getFileInfo(filename):
    word_dict = {}  # Dictionary to store words
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            word = line.strip()  # Remove extra spaces and newlines
            if word:  # Ensure it's not an empty line
                word_dict[word] = True  # Assign a default value
    
    return word_dict
def featurize_text(text):
    #split the test and create x1, x2 , 
    nouns_list= ["I", "me", "we", "us",  "my", "mine", "our", "ours", "myself", "ourselves", "you", "your", "yours", "yourself", "yourselves"]
    
    #using the picture provided in the assignment, we figure out the x's in each review
    x1 = 0
    x2 = 0
    x3 = 0
    x4 = 0
    x5 = 0
    x6 = 0
    
    x6= len(text.split(' '))
    x6 = np.round(np.log(x6), 4)

    positive_txt = getFileInfo('positive-words.txt')
    negative_txt = getFileInfo('negative-words.txt')

    for word in text.split(' '):
        if positive_txt.get(word) is not None:
            x1 += 1
        if negative_txt.get(word) is not None:
            x2 += 1
        if word == 'no':
            x3 = 1

        if word in nouns_list:
            x4 += 1
    
    for letter in text:
        if letter == '!':
            x5 = 1

    # print([x1, x2, x3, x4, x5, x6])
    return [x1, x2, x3, x4, x5, x6]

def normalize(train_vectors):


    train_vectors = np.array(train_vectors) 
    min_vals = train_vectors.min(axis=0)  #minimum
    max_vals = train_vectors.max(axis=0)  #maximim
    
    
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1  
    
    #normalization formula
    normalized_vectors = (train_vectors - min_vals) / ranges  
    # print(normalized_vectors)
    return normalized_vectors.tolist()
    