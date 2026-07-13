from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

class Model:
    def __init__(self, random_state):
        self.random_state = random_state

    def LOGO(self, X, y, groups):
        logo = LeaveOneGroupOut()
        for train_index, test_index in logo.split(X, y, groups):
            yield X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

    def train_and_evaluate(self, X, y, groups, get_sample_weights=None):
        all_metrics =[]
        for X_train, X_test, y_train, y_test in self.LOGO(X, y, groups):
            model = self.train_model()
            sample_weights = None
            if get_sample_weights is not None:
                sample_weights = get_sample_weights(y_train)
            model.fit(X_train, y_train, sample_weights)
            y_pred = model.predict(X_test)
            metrics = self.calculate_metrics(y_test, y_pred)
            all_metrics.append(metrics)
        return self.calculate_avg_metrics(all_metrics)

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

    def calculate_avg_metrics(self, all_metrics):
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = []
            for i in all_metrics:
                values.append(i[key])
            avg_metrics[key] = np.mean(values)
        return avg_metrics

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

    def plot_average_cm(self, avg_cm):
        labels = ['жировая паренхима', 'железистая', 'фиброз', 'кожа', 'артефакты']
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(avg_cm, interpolation='nearest', cmap='Blues')
        ax.set_title('average confusion matrix (5-fold CV)')
        ax.set_xlabel('pred')
        ax.set_ylabel('true')

        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels)

        thresh = avg_cm.max() / 2.
        for i in range(avg_cm.shape[0]):
            for j in range(avg_cm.shape[1]):
                ax.text(j, i, f'{avg_cm[i, j]:.1f}',
                        ha="center", va="center",
                        color="white" if avg_cm[i, j] > thresh else "black")

        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.grid(False)
        plt.show()

    def save_model(self, X, y, model_type):
        model = self.train_model()
        model.fit(X, y)
        cur_file = Path(__file__)
        model_dir = cur_file.parent.parent / 'web'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f'{model_type}.joblib'
        joblib.dump(model, model_path)