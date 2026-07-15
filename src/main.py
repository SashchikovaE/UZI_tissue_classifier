from my_random_forest import RandomForest
from my_xgboost import XGB
from preprocessing import DataPreprocessor
from draw_mask import App
from my_lightgbm import LGB

if __name__ == "__main__":
    test_image = 8
    data = DataPreprocessor('*.BMP', '*.bmp', test_image)
    X_train, X_test, y_train, y_test, coords = data.run_preprocess()

    #model = RandomForest(100, 20, 42)
    #model.run_randfor(X_train, X_test, y_train, y_test, save=True)

    model2 = XGB(200, 5, 0.1, 'multi:softmax', 5, 1.0, 0.1, 42, 'mlogloss')
    model2.run_xgb(X_train, X_test, y_train, y_test, save=False)

    model3 = LGB(42)
    model3.run_lgb(X_train, X_test, y_train, y_test, save=True)

    app = App()
    app.predict_test(test_image, X_test, coords)


