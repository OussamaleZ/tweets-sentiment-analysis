import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim.downloader as api
from keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense, GlobalAveragePooling1D
import os
import re
import gensim.downloader as api
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle


with open('X.pkl', 'rb') as f:
    X = pickle.load(f)
with open('y.pkl', 'rb') as f:
    y = pickle.load(f)
with open('X_train.pkl', 'rb') as f:
    X_train= pickle.load(f)
with open('X_test.pkl', 'rb') as f:
    X_test= pickle.load(f)
with open('y_train.pkl', 'rb') as f:
    y_train= pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test= pickle.load(f)

# Load pre-trained GloVe model
embedding_model = api.load("glove-twitter-200")  # For example, 200-dimensional GloVe embeddings

# Parameters
# Parameters
MAX_SEQUENCE_LENGTH = 10000  # Number of features/columns in X (length of each sample vector)
MAX_WORDS = len(set(word for sequence in X for word in sequence))   # Number of samples (or instances) in X
EMBEDDING_DIM = 200  # The dimension of the GloVe embeddings

#Initialize the embedding matrix with zeros
embedding_matrix = np.zeros((MAX_WORDS, EMBEDDING_DIM))

# Function to compute the average word vector for a tweet
def get_avg_embedding(tweet, model, vector_size=200):
    words = tweet.split()  # Tokenize by whitespace
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:  # If no words in the tweet are in the vocabulary, return a zero vector
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)

# Function to create word index from tokenized data (assuming X_train and X_test are already tokenized)
def create_word_index(X):
    word_index = {}
    index = 1  # Starting index for the word index (0 is reserved for padding)
    for sequence in X:
        for word in sequence:
            if word not in word_index:
                word_index[word] = index
                index += 1
    return word_index

# Create word index for tokenized data
word_index = create_word_index(X_train)
# Model components
def build_model(input_shape, embedding_matrix=None):
    # Input Layer
    input_layer = layers.Input(shape=input_shape)

    # Embedding Layer
    if embedding_matrix is not None:
        embedding_layer = Embedding(input_dim=MAX_WORDS, 
                                    output_dim=EMBEDDING_DIM, 
                                    input_length=MAX_SEQUENCE_LENGTH, 
                                    weights=[embedding_matrix], 
                                    trainable=True)(input_layer)
    else:
        embedding_layer = Embedding(input_dim=MAX_WORDS, 
                                    output_dim=EMBEDDING_DIM, 
                                    input_length=MAX_SEQUENCE_LENGTH)(input_layer)
    
    # Word-level LSTM Layer (Optional, for more complex representation)
    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)

    # Pooling Layer: Global Average Pooling
    avg_pool = GlobalAveragePooling1D()(lstm_layer)  # AVG* operation

    # Chronological LSTM Layer: LSTM on the sequence of sub-events
    chronological_lstm = LSTM(64, return_sequences=True)(lstm_layer)
    chronological_lstm = LSTM(32)(chronological_lstm)  # Final LSTM for chronological sequence

    # Dense Layer for classification
    output_layer = Dense(1, activation='sigmoid')(chronological_lstm)  # Binary classification (sub-event detection)

    # Build Model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model



X_train_pad = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
X_test_pad = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)


# Populate the embedding matrix with vectors from the pre-trained GloVe model
for word, i in word_index.items():
    if i < MAX_WORDS:
        # Directly access the word vector from the GloVe model (KeyedVectors)
        if word in embedding_model:
            embedding_matrix[i] = embedding_model[word]

# Build model
model = build_model((MAX_SEQUENCE_LENGTH,), embedding_matrix=embedding_matrix)

# Train the model
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_data=(X_test_pad, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
