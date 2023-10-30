import pandas as pd

training = pd.read_csv('training.csv', encoding_errors="ignore")
training.to_csv('training_cleaned.csv', index=False, encoding='utf-8')