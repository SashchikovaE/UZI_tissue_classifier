#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import KFold, train_test_split
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
#import numpy as np
#import matplotlib.pyplot as plt
#from pathlib import Path
#import joblib
#
#class RandomForestOriginal:
#    """
#       A wrapper class for sklearn's Random Forest classifier with enhanced evaluation capabilities.
#
#       This class provides a simplified interface for training and evaluating Random Forest models
#       with both standard train-test split and cross-validation approaches. It automatically
#       calculates multiple evaluation metrics and supports class balancing.
#
#       Attributes:
#           n_estimators (int): The number of trees in the forest.
#           max_depth (int): The maximum depth of the trees.
#           random_state (int): Random state for reproducibility.
#           test_size (float): Proportion of dataset to include in test split (0.0 to 1.0).
#           is_standard_split (bool): If True, use standard train-test split;
#                                    if False, use cross-validation.
#    """
#    def __init__(self, n_estimators, max_depth, random_state, test_size, is_standard_split):
#        """Initialize logistic regression model with specified parameters."""
#        self.n_estimators = n_estimators
#        self.max_depth = max_depth
#        self.random_state = random_state
#        self.test_size = test_size
#        self.is_standard_split = is_standard_split
#
#    def standard_split(self, X, y):
#        return train_test_split(
#            X, y, stratify=y, test_size=self.test_size, random_state=self.random_state)
#
#    def cross_validation(self, X, y):
#        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
#        for train_i, test_i in kf.split(X):
#            X_train, X_test, y_train, y_test = X.iloc[train_i], X.iloc[test_i], y.iloc[train_i], y.iloc[test_i]
#            yield (X_train, X_test, y_train, y_test)
#
#    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
#        """
#        Train model and evaluate on test set
#
#        Args:
#            X_train (DataFrame): Training features
#            y_train (Series): Training labels
#            X_test (DataFrame): Test features
#            y_test (Series): Test label
#
#        Returns:
#            dict: Evaluation metrics
#        """
#        model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state, class_weight='balanced')
#        model.fit(X_train, y_train)
#        y_pred = model.predict(X_test)
#        return self.calculate_metrics(y_test, y_pred)
#
#    def run_standard_split(self, X, y):
#        X_train, X_test, y_train, y_test = self.standard_split(X, y)
#        return self.train_and_evaluate(X_train, X_test, y_train, y_test)
#
#    def run_cross_validation(self, X, y):
#        metrics = []
#        for X_train, X_test, y_train, y_test in self.cross_validation(X, y):
#            metrics.append(self.train_and_evaluate(X_train, X_test, y_train, y_test))
#        return metrics
#
#    def calculate_metrics(self, y_test, y_pred):
#        metrics = {
#            'accuracy': round(accuracy_score(y_test, y_pred), 4),
#            'precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
#            'recall': round(recall_score(y_test, y_pred, average='weighted'), 4),
#            'f1': round(f1_score(y_test, y_pred, average='weighted'), 4),
#            #'ROC-AUC': round(roc_auc_score(y_test, y_pred, multi_class='ovr', average='weighted'), 4)
#        }
#        print(classification_report(y_test, y_pred))
#        self.conf_matrix(y_test, y_pred)
#        return metrics
#
#    def average_metrics(self, metrics):
#        metric_names = ['accuracy', 'precision', 'recall', 'f1']
#        return {
#            metric: np.mean([m[metric] for m in metrics]) for metric in metric_names
#        }
#
#    def conf_matrix(self, y_true, y_pred):
#        cm = confusion_matrix(y_true, y_pred)
#        print("Числовая матрица ошибок:\n", cm)
#        labels = ['жировая паренхима', 'железистая', 'фиброз', 'кожа', 'артефакты']
#        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#        disp.plot(cmap='Blues', values_format='d', colorbar=False)
#        plt.title('Матрица ошибок (Confusion Matrix)')
#        plt.xlabel('Предсказано моделью')
#        plt.ylabel('Истинный класс')
#        plt.grid(False)
#        plt.show()
#
#    def save_model(self, X_no3, y_no3, model_type):
#        model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state, class_weight='balanced')
#        model.fit(X_no3, y_no3)
#        #y_proba = model.predict_proba(X)
#        cur_file = Path(__file__)
#        model_dir = cur_file.parent.parent / 'web'
#        model_dir.mkdir(parents=True, exist_ok=True)
#        model_path = model_dir / f'{model_type}.joblib'
#        joblib.dump(model, model_path)
#
#    def run_randfor_orig(self, X, y, X_no3, y_no3):
#        print("sklearn random forest")
#        if self.is_standard_split:
#            print("standard split")
#            print(self.run_standard_split(X, y), "\n")
#        else:
#            print("cross validation")
#            metrics = self.run_cross_validation(X, y)
#            print(self.average_metrics(metrics))
#        self.save_model(X_no3, y_no3, 'random_forest')


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib


class RandomForestOriginal:

    def __init__(self, n_estimators, max_depth, random_state, test_size, is_standard_split):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.test_size = test_size
        self.is_standard_split = is_standard_split

    def standard_split(self, X, y):
        return train_test_split(
            X, y, stratify=y, test_size=self.test_size, random_state=self.random_state)

    def cross_validation(self, X, y):
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for train_i, test_i in kf.split(X):
            yield X.iloc[train_i], X.iloc[test_i], y.iloc[train_i], y.iloc[test_i]

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, plot_cm=True):
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        if plot_cm:
            self.conf_matrix(y_test, y_pred)
        return metrics, y_test, y_pred

    def run_standard_split(self, X, y):
        X_train, X_test, y_train, y_test = self.standard_split(X, y)
        metrics, _, _ = self.train_and_evaluate(X_train, X_test, y_train, y_test, plot_cm=True)
        return metrics

    def run_cross_validation(self, X, y):
        metrics_list = []
        cm_list = []

        for fold, (X_train, X_test, y_train, y_test) in enumerate(self.cross_validation(X, y), 1):
            fold_metrics, y_t, y_p = self.train_and_evaluate(X_train, X_test, y_train, y_test, plot_cm=False)
            metrics_list.append(fold_metrics)

            cm = confusion_matrix(y_t, y_p, labels=[1, 2, 3, 4, 5])
            cm_list.append(cm)

        avg_cm = np.mean(cm_list, axis=0)
        print("average confusion matrix (5-fold CV):")
        print(np.round(avg_cm, 2))
        self.plot_average_cm(avg_cm)

        avg_metrics = self.average_metrics(metrics_list)
        print("\naverage metrics:")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.4f}")

        return avg_metrics

    def calculate_metrics(self, y_test, y_pred):
        metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            'f1': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        }
        print(classification_report(y_test, y_pred, zero_division=0))
        return metrics

    def average_metrics(self, metrics):
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        return {metric: np.mean([m[metric] for m in metrics]) for metric in metric_names}

    def conf_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
        print("Числовая матрица ошибок:\n", cm)
        labels = ['жировая паренхима', 'железистая', 'фиброз', 'кожа', 'артефакты']
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap='Blues', values_format='d', colorbar=False)
        plt.title('confusion matrix (standard split)')
        plt.xlabel('pred')
        plt.ylabel('true')
        plt.grid(False)
        plt.tight_layout()
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

    def save_model(self, X_no3, y_no3, model_type):
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight='balanced'
        )
        model.fit(X_no3, y_no3)
        cur_file = Path(__file__)
        model_dir = cur_file.parent.parent / 'web'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f'{model_type}.joblib'
        joblib.dump(model, model_path)

    def run_randfor_orig(self, X, y, X_no3, y_no3):
        print("sklearn random forest")
        if self.is_standard_split:
            print("standard split")
            self.run_standard_split(X, y)
        else:
            print("cross validation")
            self.run_cross_validation(X, y)
        self.save_model(X_no3, y_no3, 'random_forest')