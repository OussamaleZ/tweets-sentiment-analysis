from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Dropout, TimeDistributed, Bidirectional
from keras.callbacks import EarlyStopping
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

# Parameters


MAX_SEQUENCE_LENGTH = X.shape[1]  # Number of features/columns in X (length of each sample vector)
MAX_WORDS = len(set(word for sequence in X for word in sequence))   # Number of samples (or instances) in X
EMBEDDING_DIM = 50  # Embedding dimension can remain fixed or adjusted based on your requirements
NUM_BINS = X_train.shape[1]  # Number of bins (time steps)

# Build the chronological LSTM model
model = Sequential()
# Embedding layer for word representation
model.add(Embedding(input_dim=MAX_WORDS, 
                    output_dim=EMBEDDING_DIM, 
                    input_length=MAX_SEQUENCE_LENGTH))

# Chronological LSTM for bin-level sequences
model.add(Bidirectional(LSTM(64, return_sequences=True)))  # Outputs for all time steps
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(32, return_sequences=False)))  # Outputs at the final time step
model.add(Dropout(0.3))

# Fully connected output layer for binary classification
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(optimizer="adam", 
              loss="binary_crossentropy", 
              metrics=["accuracy"])

# Train the model
early_stopping = EarlyStopping(monitor="test_loss", patience=3, restore_best_weights=True)
model.fit(X_train, 
          y_train, 
          validation_data=(X_test, y_test), 
          epochs=10, 
          batch_size=32, 
          callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")

# Save predictions for evaluation data
predictions = model.predict(X_test)
