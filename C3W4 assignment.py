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

