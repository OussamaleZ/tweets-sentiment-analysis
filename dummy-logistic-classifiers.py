from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import re
import gensim.downloader as api
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
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

# Load GloVe model with Gensim's API
embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings
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
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Function to compute the average word vector for a tweet
def get_avg_embedding(tweet, model, vector_size=200):
    words = tweet.split()  # Tokenize by whitespace
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:  # If no words in the tweet are in the vocabulary, return a zero vector
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)

# We set up a basic classifier that we train and then calculate the accuracy on our test set
clf = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Test set: ", accuracy_score(y_test, y_pred))

###### For Kaggle submission

# This time we train our classifier on the full dataset that it is available to us.
clf = LogisticRegression(random_state=42, max_iter=1000).fit(X, y)
# We add a dummy classifier for sanity purposes
dummy_clf = DummyClassifier(strategy="most_frequent").fit(X, y)

predictions = []
dummy_predictions = []
# We read each file separately, we preprocess the tweets and then use the classifier to predict the labels.
# Finally, we concatenate all predictions into a list that will eventually be concatenated and exported
# to be submitted on Kaggle.

current_directory = os.getcwd()
eval_tweets_dir = os.path.join(current_directory, "challenge_data/eval_tweets")

for fname in os.listdir(eval_tweets_dir):
    val_df = pd.read_csv(eval_tweets_dir +"/" + fname)
    val_df['Tweet'] = val_df['Tweet'].apply(preprocess_text)

    tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in val_df['Tweet']])
    tweet_df = pd.DataFrame(tweet_vectors)

    period_features = pd.concat([val_df, tweet_df], axis=1)
    period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
    period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()
    X = period_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values

    preds = clf.predict(X)
    dummy_preds = dummy_clf.predict(X)

    period_features['EventType'] = preds
    period_features['DummyEventType'] = dummy_preds

    predictions.append(period_features[['ID', 'EventType']])
    dummy_predictions.append(period_features[['ID', 'DummyEventType']])

pred_df = pd.concat(predictions)
pred_df.to_csv('logistic_predictions.csv', index=False)

pred_df = pd.concat(dummy_predictions)
pred_df.to_csv('dummy_predictions.csv', index=False)


