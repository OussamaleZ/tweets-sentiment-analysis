import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
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

# Define the binary classification model
def create_binary_classification_model(input_dim, embedding_dim=128):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=None),
        tf.keras.layers.GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary output
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Assume X_train, X_test, y_train, y_test are preprocessed
binary_model = create_binary_classification_model(input_dim=10000)  # Adjust input_dim as needed
binary_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
