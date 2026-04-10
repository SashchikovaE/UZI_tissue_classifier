from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

class LogisticRegressionC:

    def __init__(self, penalty, lambd, max_iter, class_weight,
                 random_state, test_size, is_standard_split):
        self.penalty = penalty
        self.lambd = lambd
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.test_size = test_size
        self.is_standard_split = is_standard_split

    def standard_split(self, X, y):
        return train_test_split(
            X, y, stratify=y, test_size=self.test_size, random_state=self.random_state)

    def cross_validation(self, X, y):
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for train_i, test_i in kf.split(X):
            X_train, X_test, y_train, y_test = X.iloc[train_i], X.iloc[test_i], y.iloc[train_i], y.iloc[test_i]
            yield (X_train, X_test, y_train, y_test)

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        model = LogisticRegression(penalty=self.penalty, C=1/self.lambd, solver='saga', max_iter=self.max_iter, tol=1e-5,
                                   class_weight=self.class_weight, random_state=self.random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        print("Probabilities on test data:")
        print(pd.Series(y_proba[:, 1]).describe())
        return self.calculate_metrics(y_test, y_pred)

    def run_standard_split(self, X, y):
        X_train, X_test, y_train, y_test = self.standard_split(X, y)
        return self.train_and_evaluate(X_train, X_test, y_train, y_test)

    def run_cross_validation(self, X, y):
        metrics = []
        for X_train, X_test, y_train, y_test in self.cross_validation(X, y):
            metrics.append(self.train_and_evaluate(X_train, X_test, y_train, y_test))
        return metrics

    def calculate_metrics(self, y_test, y_pred):
        metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred), 6),
            'precision': round(precision_score(y_test, y_pred, average='weighted'), 6),
            'recall': round(recall_score(y_test, y_pred, average='weighted'), 6),
            'f1': round(f1_score(y_test, y_pred, average='weighted'), 6),
            'ROC-AUC': round(roc_auc_score(y_test, y_pred, multi_class='ovr', average='weighted'), 6)
        }
        return metrics

    def average_metrics(self, metrics):
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'ROC-AUC']
        return {
            metric: np.mean([m[metric] for m in metrics]) for metric in metric_names
        }

    def save_model(self, X, y, model_type):
        model = LogisticRegression(penalty=self.penalty, C=1 / self.lambd, solver='saga', max_iter=self.max_iter, tol=1e-3,
                                class_weight=self.class_weight, random_state=self.random_state)
        model.fit(X, y)
        y_proba = model.predict_proba(X)
        cur_file = Path(__file__)
        model_dir = cur_file.parent.parent / 'api_files'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f'{model_type}.joblib'
        joblib.dump(model, model_path)

    def run_logreg(self, X, y):
        print("sklearn logistic regression")
        if self.is_standard_split:
            print("standard split")
            print(self.run_standard_split(X, y), "\n")
        else:
            print("cross validation")
            metrics = self.run_cross_validation(X, y)
            print(self.average_metrics(metrics), "\n")
        #self.save_model(X, y, 'log_regression_original')