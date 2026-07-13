from xgboost import XGBClassifier
from model import Model
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

class XGB (Model):

    def __init__(self, n_estimators, max_depth, learning_rate, objective, num_class, reg_lambda,
                 reg_alpha, random_state, eval_metric):
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.num_class = num_class
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.eval_metric = eval_metric

    def train_model(self):
        model = XGBClassifier(n_estimators=self.n_estimators,
                              max_depth=self.max_depth,
                              learning_rate=self.learning_rate,
                              objective=self.objective,
                              num_class=self.num_class,
                              reg_lambda=self.reg_lambda,
                              reg_alpha=self.reg_alpha,
                              random_state=self.random_state,
                              eval_metric=self.eval_metric)
        return model

    def get_sample_weights(self, y_train):
        class_weights = {0: 10.0, 1: 10.0, 2: 1.0, 3: 1.0, 4: 1.0}
        return np.array([class_weights[label] for label in y_train])

    def run_xgb(self, X, y, groups, save=False):
        print("sklearn XGBoost")
        print(self.train_and_evaluate(X, y, groups, get_sample_weights=self.get_sample_weights))
        if save:
            self.save_model(X, y, 'XGBoost')