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

import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceLabelingModel(nn.Module):
    def __init__(self, config):
        super(SequenceLabelingModel, self).__init__()
        
        # Configuration
        self.config = config
        
        # Embedding layer
        self.word_embeddings = nn.Embedding(
            num_embeddings=len(config['word_to_ix']), 
            embedding_dim=config['embedding_dim']
        )
        
        # Pretrained embeddings (if provided)
        if config.get('pretrained_embeddings', False):
            self.word_embeddings.weight = nn.Parameter(torch.tensor(config['pretrained_vectors'], dtype=torch.float))
        
        # LSTM for tweet representation
        self.lstm_tweet = nn.LSTM(
            input_size=config['embedding_dim'], 
            hidden_size=config['hidden_dim'], 
            batch_first=True
        )
        
        # LSTM for bin representation
        self.lstm_bin = nn.LSTM(
            input_size=config['hidden_dim'], 
            hidden_size=config['hidden_dim'], 
            batch_first=True
        )
        
        # Fully connected layer to map to tag space
        self.hidden2tag = nn.Linear(config['hidden_dim'], config['num_classes'])
        
        # Dropout layers
        self.dropout_embeddings = nn.Dropout(config['dropout_embeddings'])
        self.dropout_lstm_output = nn.Dropout(config['dropout_lstm_output'])

    def forward(self, bins, lengths):
        # Step 1: Embedding
        embedded = self.word_embeddings(bins)  # Shape: (batch_size, seq_length, embedding_dim)
        embedded = self.dropout_embeddings(embedded)
        
        # Step 2: Tweet-level representation (LSTM)
        packed_input = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm_tweet(packed_input)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Step 3: Bin-level representation (Average Pooling)
        bin_representation = torch.mean(lstm_output, dim=1)  # Average pooling over sequence length
        
        # Step 4: Sequence-level representation (Optional LSTM over bins)
        bin_representation = bin_representation.unsqueeze(0)  # Shape: (1, batch_size, hidden_dim)
        lstm_bin_output, _ = self.lstm_bin(bin_representation)
        lstm_bin_output = self.dropout_lstm_output(lstm_bin_output)
        
        # Step 5: Classification
        logits = self.hidden2tag(lstm_bin_output.squeeze(0))  # Shape: (batch_size, num_classes)
        
        return logits

from collections import defaultdict

# Assume X_train is a list of tokenized sentences, e.g., [['goal', 'yellow'], ['card', 'goal']]
word_to_ix = defaultdict(lambda: len(word_to_ix))  # Default dictionary to auto-assign indices

# Add a special padding token
word_to_ix['<PAD>'] = 0

# Populate vocabulary
for sequence in X_train:
    for word in sequence:
        word_to_ix[word]  # Assigns a new index to unseen words

# Convert defaultdict to a regular dictionary
word_to_ix = dict(word_to_ix)

# Example usage
config = {
    'word_to_ix': word_to_ix,
    'embedding_dim': 200,
    'hidden_dim': 64,
    'num_classes': 3,  # BIO tagging (B, I, O)
    'dropout_embeddings': 0.3,
    'dropout_lstm_output': 0.3,
    'pretrained_embeddings': False,  # Set True if using pretrained
    'pretrained_vectors': None  # Replace with pretrained embeddings if available
}

# Dummy data
batch_size = 4
seq_length = 10
bins = torch.randint(0, len(config['word_to_ix']), (batch_size, seq_length))  # Random token IDs
lengths = torch.randint(5, seq_length + 1, (batch_size,))  # Random sequence lengths

model = SequenceLabelingModel(config)
logits = model(bins, lengths)
print("Logits shape:", logits.shape)  # Expected: (batch_size, num_classes)

from sklearn.metrics import classification_report
import numpy as np

def evaluate_model(model, X_test, y_test, tag_to_ix, pad_token=0):
    """
    Evaluate the model on X_test and y_test.
    
    Args:
        model: Trained sequence labeling model.
        X_test: List of tokenized and padded test sentences.
        y_test: List of tokenized and padded test labels.
        tag_to_ix: Dictionary mapping tags to indices.
        pad_token: Token used for padding (default: 0).
        
    Returns:
        Classification report as a string.
    """
    ix_to_tag = {ix: tag for tag, ix in tag_to_ix.items()}  # Reverse tag mapping

    # Calculate sequence lengths (excluding padding)
    lengths = [sum(1 for token in sequence if token != pad_token) for sequence in X_test]

    # Convert to tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.long)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Forward pass with X_test and lengths
        logits = model(X_test_tensor, lengths_tensor)  
        predictions = torch.argmax(logits, dim=-1).numpy()  # Convert to class indices

    # Convert predictions and ground truth to tag labels
    y_pred = [[ix_to_tag[ix] for ix in seq] for seq in predictions]
    y_true = [[ix_to_tag[ix] for ix in seq] for seq in y_test]

    # Flatten sequences for token-level evaluation
    y_pred_flat = [tag for seq in y_pred for tag in seq]
    y_true_flat = [tag for seq in y_true for tag in seq]

    # Compute metrics
    report = classification_report(y_true_flat, y_pred_flat, digits=4)
    print(report)
    return report



# Evaluate the model
report = evaluate_model(model, X_test, y_test, word_to_ix)