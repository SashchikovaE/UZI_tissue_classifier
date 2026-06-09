import glob
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, multiscale_basic_features
from skimage.color import rgb2gray
import pandas as pd
import seaborn as sns

class DataPreprocessor():
    def __init__(self, raw, mask):
        """
            self.raw_path - расширение для сырых изображений
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
                print("good")
            else:
                i = (rgb2gray(i) * 255).astype(np.uint8)
                #i = rgb2gray(i)
            lb = local_binary_pattern(i, 8, 1)
            lbp_images.append(lb)
        return lbp_images

    def extract_lbp_patches(self, haralik_window=15, stride=4):
        lbp_images = self.lbp()
        X = []
        y = []
        half = haralik_window // 2
        for img_idx, img in enumerate(lbp_images):
            h, w = img.shape
            raw = self.raw_images[img_idx]
            mask = self.mask_images[img_idx]
            for y_coord in range(half, h - half, stride):
                for x_coord in range(half, w - half, stride):
                    patch = img[y_coord - 1:y_coord + 2, x_coord - 1:x_coord + 2]
                    features = patch.ravel().tolist()
                    row_normalized = y_coord / h
                    col_normalized = x_coord / w
                    features.append(img_idx)
                    features.append(row_normalized)
                    features.append(col_normalized)
                    rgb = tuple(mask[y_coord, x_coord])
                    target = self.classes.get(rgb, 0)
                    X.append(features)
                    y.append(target)
        return X, y

    def haralik(self, window_size=15, stride=4):
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
        half = window_size // 2
        X = []
        y = []
        coords = []
        for img_idx, raw_img in enumerate(self.raw_images):
            h, w = raw_img.shape[:2]
            mask = self.mask_images[img_idx]
            if len(raw_img.shape) == 3:
                gray = rgb2gray(raw_img) * 255
                gray = gray.astype(np.uint8)
            else:
                gray = raw_img
            for y_coord in range(half, h - half, stride):
                for x_coord in range(half, w - half, stride):
                    y_start = y_coord - half    
                    y_end = y_coord + half + 1
                    x_start = x_coord - half
                    x_end = x_coord + half + 1
                    window = gray[y_start:y_end, x_start:x_end]
                    haralik_features = self._compute_haralik_for_window(window)
                    target = self.classes.get(tuple(mask[y_coord, x_coord]), 0)
                    coords.append((y_coord, x_coord))
                    X.append(haralik_features)
                    y.append(target)
        return np.array(X), np.array(y), np.array(coords)

    def _compute_haralik_for_window(self, window):
        levels = 16
        window_scaled = (window / (256 / levels)).astype(np.uint8)
        glcm = graycomatrix(window_scaled,
                            distances=[1, 2, 3],
                            angles=[0],
                                #, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                            levels=levels,
                            symmetric=True,
                            normed=True)
        props = ['contrast', 'energy', 'homogeneity',
                 'correlation', 'dissimilarity', 'ASM']
        features = []
        for prop in props:
            feature = graycoprops(glcm, prop)
            features.extend(feature.flatten())
        return features

    def hessian_for_pixel(self, gray):
        feat = multiscale_basic_features(gray, intensity=False, edges=False, texture=True)
        return feat

    def hessian(self, haralik_window=15, stride=4):
        X = []
        y = []
        half = haralik_window // 2
        for img_idx, raw_img in enumerate(self.raw_images):
            if len(raw_img.shape) == 3:
                gray = rgb2gray(raw_img)
            else:
                gray = raw_img
            hessian_maps = self.hessian_for_pixel(gray)
            h, w, d = hessian_maps.shape
            mask = self.mask_images[img_idx]
            for y_coord in range(half, h - half, stride):
                for x_coord in range(half, w - half, stride):
                    pixel_feat = hessian_maps[y_coord, x_coord, :].tolist()
                    pixel_feat.append(y_coord / h)
                    pixel_feat.append(x_coord / w)
                    rgb = tuple(mask[y_coord, x_coord])
                    target = self.classes.get(rgb, 0)
                    X.append(pixel_feat)
                    y.append(target)
        return np.array(X), np.array(y)

    def save_csv(self, lbp, haral, gessian, y):
        if len(lbp) == len(haral) and len(lbp) == len(gessian):
            combined =[]
            for i in range(len(lbp)):
                combined_row = list(lbp[i]) + list(haral[i]) + list(gessian[i])
                combined.append(np.array(combined_row))
            df = pd.DataFrame(combined)
            df['label'] = y
            df = df[df['label'] != 0]
            #df_filtered = df[df['label'].isin([3, 4])]
            df.to_csv('db.csv', index=False)
        else:
            return 1
        return df

    def vizualize_classes_distribution(self, df, y):
        plt.figure(figsize=(10, 5))
        counts = df[y].value_counts().sort_index()
        plt.bar(counts.index.astype(str), counts.values)
        plt.title('Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.show()

    def run_prepro(self, test_image):
        self.resize()
        #self.vizualize()
        lbp, lbp_y = self.extract_lbp_patches()
        haral, haral_y, coords = self.haralik()
        gessian, gessian_y = self.hessian()
        df = self.save_csv(lbp, haral, gessian, haral_y)
        X = df.drop(['label'], axis=1)
        y = df['label']
        #self.vizualize_classes_distribution(df, 'label')
        # 9 столбец это номер картинки
        X_no_test = df[df[9] != test_image].drop(columns=['label'])
        y_no_test = df[df[9] != test_image]['label']
        X_test = df[df[9] == test_image].drop(columns=['label'])
        y_test = df[df[9] == test_image]['label']
        return X, y, X_no_test, X_test, y_no_test, y_test, np.array(coords)