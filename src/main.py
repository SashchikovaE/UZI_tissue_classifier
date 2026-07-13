from ranfor import RandomForest
from xgboosting import XGB
from preprocessing import DataPreprocessor

if __name__ == "__main__":
    test_image = 8
    data = DataPreprocessor('*.BMP', '*.bmp', test_image)
    X, y, groups, _ = data.run_prepro()

    #model = RandomForest(100, 20, 42)
    #model.run_randfor_orig(X, y, groups)

    model2 = XGB(200, 5, 0.1, 'multi:softmax', 5, 1.0, 0.1, 42, 'mlogloss')
    model2.run_xgb(X, y, groups,True)
