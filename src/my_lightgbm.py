import lightgbm as lgb
from model import Model
from sklearn.model_selection import GridSearchCV
import numpy as np

class LGB (Model):

    def __init__(self, random_state):
        super().__init__(random_state)

    def train_model(self):
        model = lgb.LGBMClassifier(
            objective='multiclass',           # тип задачи
            num_class=5,                      # число твоих классов
            metric='multi_logloss',           # метрика для оценки
            n_estimators=200,                 # кол-во деревьев
            learning_rate=0.05,               # шаг обучения
            num_leaves=38,                    # кол-во листьев
            max_depth=11,                     # макс. глубина
            class_weights = 'balanced',
            min_data_in_leaf=23,              # мин. объектов в листе
            feature_fraction=0.75,            # доля признаков
            bagging_fraction=0.70,            # доля данных для баггинга
            bagging_freq=7,                   # частота баггинга
            reg_lambda=0.04,                  # L2 регуляризация
            reg_alpha=0.2,                    # L1 регуляризация
            random_state=self.random_state,   # для воспроизводимости
        )

        return model

    def run_lgb(self, X_train, X_test, y_train, y_test, save=False):
        print("sklearn XGBoost")
        print(self.train_and_evaluate(X_train, X_test, y_train, y_test))
        if save:
            self.save_model(X_train, y_train, 'LightGBM')