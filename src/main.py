from ranfor import RandomForestOriginal
from logreg import LogisticRegressionC
from preprocessing import DataPreprocessor

if __name__ == "__main__":
    data = DataPreprocessor('*.BMP', '*.bmp')
    X, y = data.run_prepro()

    model = RandomForestOriginal(100, 20, 42, 0.3, True)
    model.run_randfor_orig(X, y)

    #model2 = LogisticRegressionC('l1', 0.1, 10000, 'balanced', 42, 0.3, True)
    #model2.run_logreg(X,y)


харалик 10или 20
гессе собств числа
номер строкики
multiscale_basic_features что это