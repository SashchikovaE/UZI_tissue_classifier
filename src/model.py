from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,\
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

class Model:
    def __init__(self, random_state):
        self.random_state = random_state

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, get_sample_weights=None):
        model = self.train_model()
        sample_weights = None
        if get_sample_weights is not None:
            sample_weights = get_sample_weights(y_train)
        model.fit(X_train, y_train, sample_weights)
        y_pred = model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        return metrics

    def train_model(self):
        raise NotImplementedError()

    def calculate_metrics(self, y_test, y_pred, plot_cm=True):
        metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred), 6),
            'precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 6),
            'recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 6),
            'f1': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 6),
        }
        print(classification_report(y_test, y_pred, zero_division=0))
        if plot_cm:
            self.conf_matrix(y_test, y_pred)
        return metrics

    def conf_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        print("Числовая матрица ошибок:\n", cm)
        labels = ['жировая паренхима', 'железистая', 'фиброз', 'кожа', 'артефакты']
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap='Blues', values_format='d', colorbar=False)
        plt.title('confusion matrix')
        plt.xlabel('pred')
        plt.ylabel('true')
        plt.grid(False)
        plt.show()

    def save_model(self, X_train, y_train, model_type):
        model = self.train_model()
        model.fit(X_train, y_train)
        model_path = Path(__file__).parent / f'{model_type}.joblib'
        joblib.dump(model, model_path)
