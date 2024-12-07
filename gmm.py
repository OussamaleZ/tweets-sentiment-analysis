from sklearn.mixture import GaussianMixture
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

# Fit a Gaussian Mixture Model for classification
n_components = len(set(y))  # Number of components should match the number of classes
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)

# Fit the GMM on the training data
gmm.fit(X_train)

# Predict class probabilities for the test set
y_pred = gmm.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {accuracy}")
