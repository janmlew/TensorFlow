import gdown
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

url = 'https://drive.google.com/uc?id=108jAePKK4R3BVYBbYJZ32JWUwxeMg20K'
out_path = 'sonnets.txt'
gdown.download(url, out_path, quiet=False)
