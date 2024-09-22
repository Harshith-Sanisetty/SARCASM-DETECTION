import pandas as pd
import numpy as np
import json
import re
import nltk
import tensorflow as tf
import zipfile
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

with zipfile.ZipFile('dataset.zip', 'r') as z:
    with z.open('dataset.json') as file:
        data = [json.loads(line) for line in file]

headlines = pd.DataFrame(data)
headlines = headlines[['headline', 'is_sarcastic']]

def preprocess_text(text):
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

headlines['cleaned_headline'] = headlines['headline'].apply(preprocess_text)

max_words = 10000
max_len = 30
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(headlines['cleaned_headline'])
sequences = tokenizer.texts_to_sequences(headlines['cleaned_headline'])
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(data, headlines['is_sarcastic'], test_size=0.2, stratify=headlines['is_sarcastic'], random_state=42)

model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_len))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=64, callbacks=[early_stop])

y_pred = (model.predict(X_test) > 0.5).astype("int32")

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Sarcastic', 'Sarcastic'], yticklabels=['Not Sarcastic', 'Sarcastic'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
