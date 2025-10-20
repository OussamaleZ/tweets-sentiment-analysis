import argparse
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


MODELS = {
    "logreg": LogisticRegression(max_iter=5000, random_state=42),
    "svm": SVC(random_state=42),
    "dt": DecisionTreeClassifier(random_state=42),
    "rf": RandomForestClassifier(random_state=42),
    "mlp": MLPClassifier(max_iter=2000, random_state=42),
    "xgb": XGBClassifier(random_state=42),
}

GRIDS = {
    "logreg": {"C": [0.01, 0.1, 1.0, 10, 30, 50, 100], "solver": ["lbfgs", "liblinear", "saga"]},
    "svm": {"C": [0.01, 0.1, 1.0, 10, 100], "kernel": ["linear", "rbf", "poly", "sigmoid"], "gamma": ["scale", "auto"]},
    "dt": {"criterion": ["gini", "entropy", "log_loss"], "max_depth": [None, 10, 20, 30, 40], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]},
    "rf": {"n_estimators": [50, 100, 200], "max_features": ["sqrt"], "min_samples_split": [2], "min_samples_leaf": [1], "bootstrap": [True]},
    "mlp": {"hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)], "activation": ["tanh", "relu"], "solver": ["adam", "sgd"], "alpha": [0.0001, 0.001, 0.01]},
    "xgb": {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7, 10], "learning_rate": [0.01, 0.1, 0.2], "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "gamma": [0, 0.1, 0.3]},
}


def load_data(data_dir: str):
    with open(os.path.join(data_dir, "X_train.pkl"), "rb") as f:
        X_train = pickle.load(f)
    with open(os.path.join(data_dir, "y_train.pkl"), "rb") as f:
        y_train = pickle.load(f)
    with open(os.path.join(data_dir, "X_test.pkl"), "rb") as f:
        X_test = pickle.load(f)
    with open(os.path.join(data_dir, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)
    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser(description="Train classic models with optional GridSearchCV, reflecting final_notebook experiments")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True)
    parser.add_argument("--grid", action="store_true", help="Use GridSearchCV with predefined grid")
    parser.add_argument("--cv", type=int, default=5, help="CV folds for grid search")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data(args.data_dir)
    model = MODELS[args.model]

    if args.grid:
        grid = GRIDS[args.model]
        search = GridSearchCV(estimator=model, param_grid=grid, cv=args.cv, scoring="accuracy", verbose=1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        y_pred = best.predict(X_test)
        print("Best params:", search.best_params_)
        print("Test accuracy:", accuracy_score(y_test, y_pred))
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Test accuracy:", accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()


