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

countries = {
    "argentina", "brazil", "france", "germany", 
    "netherlands", "usa", "croatia", "chile", "belgium", "australia",
    "spain","cameroun","brazil","united","states","honduras","switzerland",
    "croatia","mexico","netherlands","portugal","ghana","slovenia",
    "greece","ivory","coast","algeria","serbia","nigeria","south","korea","southkorea",
}

custom_stopwords = {
    "fifaworldcup", "world","fifa"
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
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", '', text)    
    # Remove punctuation
    # text = re.sub(r'[^\w\s]', '', text)
    #Replace " vs " with ""
    text = re.sub(r"\b vs \b", "", text)
    # Remove numbers
    # text = re.sub(r'\d+', '', text) 
    # Remove 'RT' at the beginning
    # text = re.sub(r'^rt\s+', '', text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", '', text)  
    # Remove elongated words
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # Replace '  ' by ' ' (Remove multiple spaces)
    text = re.sub(r' +', ' ', text)
    #Remove countries with the string ""
    # text = re.sub(r'\b(?:{})\b'.format('|'.join(countries)), '', text)
    # Replace single-letter words with an empty string
    pattern = r"\b[a-zA-Z]\b"
    # text = re.sub(pattern, "", text)
    # Remove extra spaces that might be left behind
    text = re.sub(r"\s+", " ", text).strip()
    #Replace "im" with "I am"
    text = re.sub(r"\bim\b", "I am", text)
    
    # Tokenization
    words = text.split()  

    # words = [word for word in words if word not in stop_words]
    # words = [word for word in words if word not in custom_stopwords]
    # Lemmatization
    # lemmatizer = WordNetLemmatizer()
    # words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)


def preprocess():
    # Ensure the output directory exists
    output_dir = "eval_tweets_preprocessed_soft"
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

print(preprocess_text("RT @_theworldcup: If Germany win tonight we will give this GOLD iPhone5 away to 1 lucky winner just RT & FOLLOW @_theworldcup to enter http…"))
preprocess()