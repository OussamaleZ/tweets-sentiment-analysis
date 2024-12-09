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



# Download some NLP models for processing, optional
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english')) - {"not", "never", "no"}
custom_stopwords = {
    "fifaworldcup", "argentina", "brazil", "france", "germany", 
    "netherlands", "usa", "croatia", "chile", "belgium", "australia",
    "spain","cameroun","brazil","united","states","honduras","switzerland",
    "croatia","mexico","netherlands","portugal","ghana","slovenia",
    "greece","ivory","coast","world","worldcupfinal","worldcup","cup"
}


# Extract usernames from the 'Tweet' column
def extract_usernames(tweet):
    # Match the first occurrence of a username (e.g., @username)
    match = re.search(r'@\w+', tweet)
    return match.group(0) if match else None



def preprocess_text(text):
    # Lowercasing
    text = str(text).lower()    
    # Remove hashtags    
    text = re.sub(r"#\w+", '', text)
    # Remove mentions
    text = re.sub(r"@\w+", '', text) 
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text) 
    # Remove numbers
    text = re.sub(r'\d+', '', text) 
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", '', text)     
    # Step 1: Remove 'RT' at the beginning
    text = re.sub(r'^rt\s+', '', text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", '', text)  
    # Remove elongated words
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Tokenization
    words = text.split()  

    words = [word for word in words if word not in stop_words]
    words = [word for word in words if word not in custom_stopwords]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)


def preprocess():
    # Ensure the output directory exists
    output_dir = "eval_tweets_preprocessed"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all files in the eval_tweets directory
    for filename in os.listdir("eval_tweets"):
        print("preprocessing", filename)
        # Read the file
        file_path = os.path.join("eval_tweets", filename)
        df = pd.read_csv(file_path)
        
        #Lowercase 
        df['Tweet'] = df['Tweet'].str.lower()

        #Remove the spammers
        df['Username'] = df['Tweet'].apply(extract_usernames)
        username_counts = df['Username'].value_counts()
        spammer_usernames = username_counts[username_counts > 20].index
        df = df[~df['Username'].isin(spammer_usernames)]
        df = df.drop(columns=['Username'])


        # Apply preprocessing to each tweet
        df['Tweet'] = df['Tweet'].apply(preprocess_text)
        
        # Save the preprocessed DataFrame to the new directory
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)

print(preprocess_text("RT @NASA: Learning about aerodynamics is our GOOOOOOAAAL! Check out our tests of the #WorldCup ball: http://t.co/HgGv1FDRFf http://t.co/9fn…"))
preprocess()