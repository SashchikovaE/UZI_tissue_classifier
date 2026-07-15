from sklearn.ensemble import RandomForestClassifier
from model import Model

class RandomForest (Model):

    def __init__(self, n_estimators, max_depth, random_state):
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def train_model(self):
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight={0: 10.0, 1: 10.0, 2: 1.0, 3: 1.0, 4: 1.0}
        )
        return model

    def run_randfor(self, X_train, X_test, y_train, y_test, save=False):
        print("sklearn random forest")
        print(self.train_and_evaluate(X_train, X_test, y_train, y_test))
        if save:
            self.save_model(X_train, y_train, 'random_forest')