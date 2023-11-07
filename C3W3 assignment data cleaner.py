import pandas as pd
import os
import string

# training = pd.read_csv('training.csv', encoding_errors="ignore")
# training.to_csv('training_cleaned.csv', index=False, encoding='utf-8')

if os.path.exists("glove.6B.100d.txt"):
    os.remove("glove.6B.100d.txt")

with open('glove_unclean.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Usuń znaki spoza zakresu liter łacińskich
text = ''.join(filter(lambda x: x in string.printable, text))

# Zamień znaki diakrytyczne na ich odpowiedniki bez znaków diakrytycznych
# text = text.translate(str.maketrans('ąćęłńóśźżĄĆĘŁŃÓŚŹŻ', 'acelnoszzACELNOSZZ'))

with open('glove.6B.100d.txt', 'w', encoding='utf-8') as f:
    f.write(text)

'''
with open('glove_unclean.txt', 'rb') as f:
    with open('glove.6B.100d.txt', 'w', errors='ignore') as textfile:
        textfile.write(f.read().decode('utf-8', errors='ignore'))
'''
