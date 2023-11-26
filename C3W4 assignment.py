import gdown
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

url = 'https://drive.google.com/uc?id=108jAePKK4R3BVYBbYJZ32JWUwxeMg20K'
SONNETS_FILE = 'sonnets.txt'
gdown.download(url, SONNETS_FILE, quiet=False)

with open(SONNETS_FILE) as f:
    data = f.read()

corpus = data.lower().split("\n")

print(f"There are {len(corpus)} lines of sonnets\n")
print(f"The first 5 lines look like this:\n")
for i in range(5):
    print(corpus[i])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(corpus[0])

print(tokenizer.texts_to_sequences(corpus[0]))
print(tokenizer.texts_to_sequences([corpus[0]]))
print(tokenizer.texts_to_sequences([corpus[0]])[0])


def n_gram_seqs(corpus, tokenizer):
    """
    Generates a list of n-gram sequences
    Args:
        corpus (list of string): lines of texts to generate n-grams for
        tokenizer (object): an instance of the Tokenizer class containing the word-index dictionary
    Returns:
        input_sequences (list of int): the n-gram sequences for each line in the corpus
    """
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
    return input_sequences


# Tests the function with one example
first_example_sequence = n_gram_seqs([corpus[0]], tokenizer)
print("n_gram sequences for first example look like this:\n")
print(first_example_sequence)

# Tests the function with a bigger corpus
next_3_examples_sequence = n_gram_seqs(corpus[1:4], tokenizer)
print("n_gram sequences for next 3 examples look like this:\n")
print(next_3_examples_sequence)