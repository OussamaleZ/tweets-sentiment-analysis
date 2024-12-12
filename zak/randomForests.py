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

# Function to compute the average word vector for a tweet
def get_avg_embedding(tweet, model, vector_size=200):
    words = tweet.split()  # Tokenize by whitespace
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:  # If no words in the tweet are in the vocabulary, return a zero vector
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)
# Load GloVe model with Gensim's API
embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings
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

# Fit a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score
# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Initialize the Random Forest model with additional hyperparameters
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=15, 
    min_samples_split=10, 
    min_samples_leaf=5, 
    max_features='sqrt', 
    random_state=42, 
    class_weight='balanced'
)

# Train the model
rf_model.fit(X_train, y_train)

# Evaluate the model with cross-validation
cross_val_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {cross_val_scores.mean():.4f} ± {cross_val_scores.std():.4f}")

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"RF Test Accuracy: {accuracy:.4f}")

current_directory = os.getcwd()
eval_tweets_dir = os.path.join(current_directory, "challenge_data/eval_tweets_preprocessed")

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

    preds = rf_model.predict(X)

    period_features['EventType'] = preds


    predictions.append(period_features[['ID', 'EventType']])

pred_df = pd.concat(predictions)
pred_df.to_csv('rf.csv', index=False)


