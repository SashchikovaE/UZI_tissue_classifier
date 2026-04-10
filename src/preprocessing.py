import glob
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray
import pandas as pd

class DataPreprocessor():
    def __init__(self, raw, mask):
        """
            self.raw_path - расширение для сырых изо
            self.mask_path - расширение для масок
            self.raw_images - массивы картинок
            self.mask_images -
        """
        self.raw_path = raw
        self.mask_path = mask
        self.raw_images = []
        self.mask_images = []
        self.load()
        self.classes = {
            (0, 0, 0): 0,  # черный → фон
            (255, 255, 0): 1,  # желтый → жировая паренхима
            (0, 255, 255): 2,  # голубой → железистая
            (0, 255, 0): 3,  # зеленый → фиброз
            (255, 0, 0): 4,  # красный → кожа
            (255, 0, 255): 5  # фиолетовый → артефакты
        }

    def load(self):
        """
            получает отсортированный массив картинок в виде названий (string)
            и записывает каждую в массив картинок(атрибут класса)
            :return:
            массив картинок
        """
        raw_path_list = sorted(glob.glob(str(Path(__file__).parent.parent / 'data/raws' / self.raw_path)))
        mask_path_list = sorted(glob.glob(str(Path(__file__).parent.parent / 'data/masks' / self.mask_path)))
        for r, m in zip(raw_path_list, mask_path_list):
            raw = imread(r)
            mask = imread(m)
            self.raw_images.append(raw)
            self.mask_images.append(mask)
        print("loaded!")
        return self.raw_images, self.mask_images

    def vizualize(self):
        for i in range(9):
            plt.subplot(2, 9, i + 1)
            plt.axis("off")
            plt.imshow(self.raw_images[i])
            plt.subplot(2, 9, i + 10)
            plt.axis("off")
            plt.imshow(self.mask_images[i])
        plt.show()

    def resize(self):
        """
            resize(image, output_shape, order=1, preserve_range=False,
            anti_aliasing=True, mode='reflect')
            у каждого пикселя есть цвет под опред номером.
            Красный = (255, 0, 0), Зеленый = (0, 255, 0), Синий = (0, 0, 255)
            Черный = (0), Серый (128), Белый (255)
            - order = 0 для классов (берет ближайшего соседа, чтобы было целое число)
            умножаем координату старой размерности на то число, во сколько раз увеличится новая размерность.
            и цвет получвшейся координаты на старой поверхности и будет в новой.
            - order = 1 билейно ()
            - order = 2 биквадрат ()
            - order = 3 бикуб ()
            - preserve_range -
            - anti_aliasing - контрастность. фолс-пискельность четкость. тру-размытие плавный переход.
            - mode - 'reflect'
            :return:
        """
        raw_resized = []
        mask_resized = []
        for r, m in zip(self.raw_images, self.mask_images):
            ra = resize(r,(256, 256), preserve_range=True) / 255.0
            ma = resize(m,(256, 256), preserve_range=True, order=0)
            raw_resized.append(ra)
            mask_resized.append(ma)
        self.raw_images = raw_resized
        self.mask_images = mask_resized

    def lbp(self):
        lbp_images = []
        for i in self.raw_images:
            if len(i.shape) < 3:
                print("goooood")
            else:
                i = rgb2gray(i)
            lb = local_binary_pattern(i, 8, 1)
            lbp_images.append(lb)
        return lbp_images

    def extract_patches(self):
        lbp_images = self.lbp()
        X = []
        y = []
        for img_idx, img in enumerate(lbp_images):
            h, w = img.shape
            raw = self.raw_images[img_idx]
            mask = self.mask_images[img_idx]
            for y_coord in range(1, h - 1):
                for x_coord in range(1, w - 1):
                    patch = img[y_coord - 1:y_coord + 2, x_coord - 1:x_coord + 2]
                    features = patch.ravel()
                    row_normalized = y_coord / h
                    features.append(row_normalized)
                    rgb = tuple(mask[y_coord, x_coord])
                    target = self.classes.get(rgb, 0)
                    X.append(features)
                    y.append(target)
        df = pd.DataFrame(X)
        df['label'] = y
        print(df['label'].value_counts())
        df.to_csv('db.csv', index=False)
        return df

    def haralic(self):
        """
        glcm = graycomatrix(image, distances, angles, levels,
                    symmetric=False, normed=False)
        image - 2д картинка
        distances - дистанция сравниваемых пикселей
        angles - угол
        levels -
        symmetric тру - пары i=j и j=i считаются одинаковыми. не учитывается порядок
        normed тру - считает вероятность встречи одного значения с другим, а фолз просто счетчик
        """
        features = []
        for img in self.raw_images:
            if len(img.shape) == 3:
                gray = rgb2gray(img) * 255
            gray = gray.astype(np.uint8)
            glcm = graycomatrix(gray,
                                distances=[1, 2, 3],
                                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                levels=256,
                                symmetric=True,
                                normed=True)
            # 6 основных признаков
            props = ['contrast', 'energy', 'homogeneity',
                     'correlation', 'dissimilarity', 'ASM']
            img_features = []
            for prop in props:
                feature = graycoprops(glcm, prop)
                img_features.extend(feature.flatten())
            features.append(img_features)
        return np.array(features)

    #def vizualize_correlation_matrix(self):
    #def vizualize_classes_distribution(self):

    def run_prepro(self):
        self.resize()
        self.vizualize()
        print(self.lbp()[0])
        df = self.extract_patches()
        return df.drop(['label'], axis=1), df['label']
