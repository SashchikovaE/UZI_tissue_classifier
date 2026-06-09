import sys
from pathlib import Path
import joblib
import numpy as np
import cv2
from skimage.io import imsave
current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent
src_path = parent_dir / 'src'
sys.path.append(str(src_path))
from skimage.restoration import denoise_tv_chambolle
from preprocessing import DataPreprocessor

class App():
    def __init__(self):
        self.model = self.load_model()
        self.classes = {
            0: (0, 0, 0),      # фон
            1: (255, 255, 0),  # жировая паренхима
            2: (0, 255, 255),  # железистая
            3: (0, 255, 0),    # фиброз
            4: (255, 0, 0),    # кожа
            5: (255, 0, 255)   # артефакты
        }

    def load_model(self):
        cur_file = Path(__file__)
        model_dir = cur_file.parent
        model_path = model_dir / 'random_forest.joblib'
        if not model_path.exists():
            raise FileNotFoundError(f"file of model is not found: {model_path}")
        if model_path.stat().st_size == 0:
            raise ValueError(f"file of model is empty: {model_path}")
        model = joblib.load(model_path)
        return model

    def preprocess(self, test_image):
        data = DataPreprocessor('*.BMP', '*.bmp')
        return data.run_prepro(test_image)

    def compute_metrics(self, mask_pred, mask_true, n_classes=6):
        metrics = {}
        for cls in range(n_classes):
            pred_bin = (mask_pred == cls)
            true_bin = (mask_true == cls)
            intersection = np.logical_and(pred_bin, true_bin).sum()
            union = np.logical_or(pred_bin, true_bin).sum()
            total = pred_bin.sum() + true_bin.sum()
            iou = intersection / union if union > 0 else np.nan
            dice = 2 * intersection / total if total > 0 else np.nan
            metrics[f'class_{cls}_iou'] = iou
            metrics[f'class_{cls}_dice'] = dice
        metrics['mean_iou'] = np.nanmean([v for k, v in metrics.items() if 'iou' in k and 'class' in k])
        metrics['mean_dice'] = np.nanmean([v for k, v in metrics.items() if 'dice' in k and 'class' in k])
        return metrics

    def _fill_sparse_mask(self, sparse_mask, coords, preds, stride=4):
        """Заполняет разреженную маску: каждое предсказание расширяется на блок stride×stride"""
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

    def predict(self, test_image, img_size=(256, 256), stride=4):
        X, y, X_no_test, X_test, y_no_test, y_test, coords_3 = self.preprocess(test_image)
        preds = self.model.predict(X_test)
        h, w = img_size
        mask_2d = np.zeros((h, w), dtype=np.uint8)
        mask_2d = self._fill_sparse_mask(mask_2d, coords_3, preds, stride=stride)
#
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in self.classes.items():
            rgb_mask[mask_2d == cls_id] = color
        rgb_mask = cv2.resize(rgb_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        out_path = Path(__file__).parent / "prediction_mask_img3f.png"
        imsave(out_path, rgb_mask)
        mask_true_2d = np.zeros((h, w), dtype=np.uint8)
        mask_true_2d = self._fill_sparse_mask(mask_true_2d, coords_3, y_test, stride=stride)
        #print(self.compute_metrics(mask_2d, mask_true_2d))
        #print(self.compute_metrics(mask_2d, y_3))
        return rgb_mask


    #def predict(self, test_image, img_size=(256, 256), stride=4):
    #    X, y, X_no_test, X_test, y_no_test, y_test, coords_3 = self.preprocess(test_image)
    #    preds = self.model.predict(X_test)
    #    h, w = img_size
    #    mask_2d = np.zeros((h, w), dtype=np.uint8)
    #    mask_2d = self._fill_sparse_mask(mask_2d, coords_3, preds, stride=stride)
#
    #    # TV filter - работает с мультиклассом
    #    mask_float = mask_2d.astype(np.float32)
    #    mask_denoised = denoise_tv_chambolle(mask_float, weight=0.1)
    #    mask_final = np.round(mask_denoised).astype(np.uint8)
#
    #    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    #    for cls_id, color in self.classes.items():
    #        rgb_mask[mask_final == cls_id] = color
#
    #    rgb_mask = cv2.resize(rgb_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    #    out_path = Path(__file__).parent / "prediction_mask_img9.png"
    #    imsave(out_path, rgb_mask)
#
    #    return rgb_mask

if __name__ == "__main__":
    app = App()
    test_image = 2
    print(app.predict(test_image))
