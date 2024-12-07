from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import accuracy_score

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

# Train the Support Vector Machine
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Test Accuracy: {accuracy}")
