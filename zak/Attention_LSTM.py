import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, LayerNormalization, Attention, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os
import pandas as pd
import re
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

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


# Reshape inputs for attention (batch_size, seq_length, feature_dim)
newX = X[:, np.newaxis, :]
X_train = X_train[:, np.newaxis, :]  # Add a sequence dimension
X_test = X_test[:, np.newaxis, :]

# Define Word Attention Layer (Non-Chronological Relaxed Attention)
class WordAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(WordAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create learnable parameters for the attention mechanism
        self.attention_weights = self.add_weight(
            name="attention_weights", shape=(input_shape[-1],), initializer="uniform", trainable=True
        )
        super(WordAttention, self).build(input_shape)

    def call(self, inputs):
        # Calculate attention weights for each input
        attention_scores = tf.reduce_sum(inputs * self.attention_weights, axis=-1, keepdims=True)
        attention_probs = tf.nn.softmax(attention_scores, axis=1)
        weighted_input = inputs * attention_probs
        return weighted_input

# Define the model with Word Attention and LSTM
input_layer = Input(shape=(1, 200))  # Sequence length is 1, embedding size is 200

# Apply Word Attention
attended_input = WordAttention()(input_layer)

# LSTM layer
lstm_output = LSTM(64, return_sequences=True, use_bias=True, recurrent_initializer='glorot_uniform', kernel_initializer='glorot_uniform', unroll=True)(attended_input)

# Apply Global Average Pooling
pooled_output = GlobalAveragePooling1D()(lstm_output)

# Dense layers for classification
dense_layer = Dense(64, activation="relu")(pooled_output)
dropout_layer = Dropout(0.2)(dense_layer)
output_layer = Dense(1, activation="sigmoid")(dropout_layer)

# Compile the model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(newX, y, epochs=17, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
predictions = model.predict(X_test)
threshold = 0.65
predicted_classes = (predictions > threshold).astype(int)

# Evaluate predictions
print(f"Classification Report for threshold {threshold}:")
print(classification_report(y_test, predicted_classes))
print(f"ROC AUC Score: {roc_auc_score(y_test, predicted_classes)}")

# Load GloVe model with Gensim's API
embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings

# Function to compute the average word vector for a tweet
def get_avg_embedding(tweet, model, vector_size=200):
    words = tweet.split()  # Tokenize by whitespace
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:  # If no words in the tweet are in the vocabulary, return a zero vector
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)

stop_words = set(stopwords.words('english'))
# Basic preprocessing function
def preprocess_text(text):
    # Lowercasing
    text = str(text).lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenization
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)



current_directory = os.getcwd()
eval_tweets_dir = os.path.join(current_directory, "challenge_data/eval_tweets")

# Apply preprocessing to each tweet and obtain vectors
vector_size = 200  # Adjust based on the chosen GloVe model
predictions =[]
for fname in os.listdir(eval_tweets_dir):
    val_df = pd.read_csv(eval_tweets_dir +"/" + fname)
    val_df['Tweet'] = val_df['Tweet'].apply(preprocess_text)

    tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in val_df['Tweet']])
    tweet_df = pd.DataFrame(tweet_vectors)

    period_features = pd.concat([val_df, tweet_df], axis=1)
    period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
    period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()
    X = period_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values

    X = X.reshape((-1, 1, 200)) 
    preds = model.predict(X)
    predicted_classes = (preds > threshold).astype(int)
    period_features['EventType'] = predicted_classes


    predictions.append(period_features[['ID', 'EventType']])

pred_df = pd.concat(predictions)
pred_df.to_csv('word_att.csv', index=False)
