from ranfor import RandomForestOriginal
from logreg import LogisticRegressionC
from preprocessing import DataPreprocessor

if __name__ == "__main__":
    data = DataPreprocessor('*.BMP', '*.bmp')
    test_image = 2
    X, y, X_no_test, X_test, y_no_test, y_test, coords = data.run_prepro(test_image)

    model = RandomForestOriginal(100, 20, 42, 0.3, False)
    model.run_randfor_orig(X, y, X_no_test, y_no_test)

    #model2 = LogisticRegressionC('l1', 0.1, 10000, 'balanced', 42, 0.3, False)
    #model2.run_logreg(X, y, X_no_test, y_no_test)
