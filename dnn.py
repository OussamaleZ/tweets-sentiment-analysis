from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle
import math

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
# Build a simple DNN model
dnn_model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])


dnn_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import LearningRateScheduler

# Learning rate scheduler
initial_learning_rate = 0.01
lr_schedule = LearningRateScheduler(lambda epoch: initial_learning_rate * math.exp(-0.1 * epoch))

dnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[lr_schedule])


# Evaluate the model
loss, accuracy = dnn_model.evaluate(X_test, y_test)
print(f"DNN Test Accuracy: {accuracy}")
