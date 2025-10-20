import argparse
import os
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping


def main():
    parser = argparse.ArgumentParser(description="Train a simple LSTM on precomputed embeddings")
    parser.add_argument("--data-dir", required=True, help="Directory containing processed pickles (X.pkl, y.pkl, X_train.pkl, ...)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "X_train.pkl"), "rb") as f:
        X_train = pickle.load(f)
    with open(os.path.join(args.data_dir, "y_train.pkl"), "rb") as f:
        y_train = pickle.load(f)
    with open(os.path.join(args.data_dir, "X_test.pkl"), "rb") as f:
        X_test = pickle.load(f)
    with open(os.path.join(args.data_dir, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)

    # If X are averaged embedding vectors, we'd use a small feed-forward net
    # For illustration, we reshape to sequences of length 1 with feature dim
    X_train_seq = np.expand_dims(X_train, axis=1)
    X_test_seq = np.expand_dims(X_test, axis=1)

    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(X_train_seq, y_train, validation_data=(X_test_seq, y_test), epochs=args.epochs, batch_size=args.batch_size, callbacks=[early_stopping])

    loss, accuracy = model.evaluate(X_test_seq, y_test)
    print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")


if __name__ == "__main__":
    main()


