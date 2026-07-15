import sys
from pathlib import Path
import joblib
import numpy as np
from skimage.io import imsave

current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent
src_path = parent_dir / 'src'
sys.path.append(str(src_path))

from preprocessing import DataPreprocessor


class App:
    def __init__(self, ):
        self.model = self.load_model()
        self.classes = {
            0: (0, 0, 0),  # фон
            1: (255, 255, 0),  # жировая
            2: (0, 255, 255),  # железистая
            3: (0, 255, 0),  # фиброз
            4: (255, 0, 0),  # кожа
            5: (255, 0, 255)  # артефакты
        }

    def load_model(self):
        model_path = Path(__file__).parent / 'LightGBM.joblib'
        if not model_path.exists():
            raise FileNotFoundError(f"file of model is not found: {model_path}")
        if model_path.stat().st_size == 0:
            raise ValueError(f"file of model is empty: {model_path}")
        model = joblib.load(model_path)
        return model

    def fill_sparse_mask(self, sparse_mask, coords, preds, stride=4):
        h, w = sparse_mask.shape
        filled = np.zeros((h, w), dtype=np.uint8)
        for (y, x), pred in zip(coords, preds):
            y, x = int(y), int(x)
            y_start = max(0, y - stride // 2)
            y_end = min(h, y + stride // 2 + 1)
            x_start = max(0, x - stride // 2)
            x_end = min(w, x + stride // 2 + 1)
            filled[y_start:y_end, x_start:x_end] = int(pred)
        return filled

    def predict_test(self, test_image, X_test, coords, img_size=(256, 256), stride=4):
        preds = self.model.predict(X_test)
        preds = preds + 1
        h, w = img_size
        mask_2d = np.zeros((h, w), dtype=np.uint8)
        mask_2d = self.fill_sparse_mask(mask_2d, coords, preds, stride=stride)

        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in self.classes.items():
            rgb_mask[mask_2d == cls_id] = color
        out_path = Path(__file__).parent / f"prediction_mask_img_{test_image + 1}.png"
        imsave(out_path, rgb_mask)
        return rgb_mask
