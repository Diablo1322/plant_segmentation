from ultralytics import YOLO
import cv2
import numpy as np
import json
from pathlib import Path
from skimage import morphology


class PlantMeasurer:
    def __init__(self, model_path, calibration_file):
        # загружаем модель
        self.model = YOLO(model_path)

        # загружаем калибровку
        with open(calibration_file, 'r') as f:
            self.calibration = json.load(f)

        self.pixels_per_mm = self.calibration['pixels_per_mm']
        self.mm_per_pixel = 1.0 / self.pixels_per_mm
        self.area_per_pixel_mm2 = self.mm_per_pixel ** 2

        # целевой размер как в Roboflow
        self.target_size = 640

    def preprocess_like_roboflow(self, image):
        """
        Предобработка изображения как в Roboflow:
        - Stretch to 640x640 (как на скриншоте)
        - Сохранение цветности
        """
        if image is None:
            return None

        # растягиваем до 640x640 (как в Roboflow)
        resized = cv2.resize(image, (self.target_size, self.target_size))

        return resized

    def measure_length(self, mask):
        # измеряет длину по скелету маски
        if mask is None or mask.size == 0:
            return 0.0
        try:
            skeleton = morphology.skeletonize(mask)
            length_pixels = np.sum(skeleton)
            # корректируем масштаб с учётом изменения размера
            scale_correction = self.target_size / 320  # если в калибровке было 320
            return length_pixels * self.mm_per_pixel * scale_correction
        except:
            return 0.0

    def measure_area(self, mask):
        # измеряет площадь маски
        if mask is None or mask.size == 0:
            return 0.0
        try:
            area_pixels = np.sum(mask > 0)
            # корректируем масштаб с учётом изменения размера
            scale_correction = (self.target_size / 320) ** 2
            return area_pixels * self.area_per_pixel_mm2 * scale_correction
        except:
            return 0.0

    def process_image(self, image_path, save_visualization=True, conf=0.3):
        # загружаем изображение
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        # СОХРАНЯЕМ ОРИГИНАЛ для калибровки
        original_h, original_w = img.shape[:2]

        # ПРЕДОБРАБОТКА КАК В ROBOFLOW
        processed_img = self.preprocess_like_roboflow(img)

        # переводим в RGB для YOLO
        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, conf=conf)[0]

        # если ничего не найдено
        if results.masks is None:
            return {
                'image_name': Path(image_path).name,
                'plants': [],
                'total_plants': 0,
                'leaf_count': 0,
                'total_leaves_area_mm2': 0.0
            }

        # получаем все маски и классы
        masks = []
        classes = []

        for mask, cls_id in zip(results.masks.data, results.boxes.cls):
            cls_name = results.names[int(cls_id.item())]
            mask_np = mask.cpu().numpy().astype(np.uint8)

            # маски уже в размере 640x640, их не меняем
            masks.append(mask_np)
            classes.append(cls_name)

        # Определяем тип растения
        plant_type = 'unknown'
        for cls_name in classes:
            if 'wheat' in cls_name:
                plant_type = 'wheat'
                break
            elif 'rocket' in cls_name:
                plant_type = 'rocket'
                break

        # Разделяем по типам
        root_masks = []
        stem_masks = []
        leaf_masks = []

        for i, cls_name in enumerate(classes):
            if 'root' in cls_name:
                root_masks.append(masks[i])
            elif 'stem' in cls_name:
                stem_masks.append(masks[i])
            elif 'leaf' in cls_name:
                leaf_masks.append(masks[i])

        # Формируем данные о растении
        plant_info = {
            'plant_id': 1,
            'type': plant_type,
            'root': None,
            'stem': None,
            'leaves': [],
            'leaf_count': len(leaf_masks),
            'total_leaves_area_mm2': 0.0
        }

        # Добавляем корень ТОЛЬКО если он есть
        if root_masks:
            plant_info['root'] = {
                'length_mm': float(self.measure_length(root_masks[0])),
                'area_mm2': float(self.measure_area(root_masks[0]))
            }

        # Добавляем стебель ТОЛЬКО если он есть
        if stem_masks:
            plant_info['stem'] = {
                'length_mm': float(self.measure_length(stem_masks[0])),
                'area_mm2': float(self.measure_area(stem_masks[0]))
            }

        # Добавляем листья
        leaf_area_sum = 0
        for j, leaf_mask in enumerate(leaf_masks):
            leaf_area = self.measure_area(leaf_mask)
            leaf_area_sum += leaf_area
            plant_info['leaves'].append({
                'leaf_id': j + 1,
                'area_mm2': float(leaf_area)
            })

        plant_info['total_leaves_area_mm2'] = float(leaf_area_sum)

        # Визуализация на ОРИГИНАЛЬНОМ размере
        vis_img = None
        if save_visualization:
            vis_img = img.copy()  # используем оригинал для визуализации

        if save_visualization and vis_img is not None:
            # Рисуем корни (нужно масштабировать маски обратно к оригинальному размеру)
            scale_x = original_w / self.target_size
            scale_y = original_h / self.target_size

            for root_mask in root_masks:
                try:
                    # масштабируем маску к оригинальному размеру
                    if root_mask.size > 0:
                        scaled_mask = cv2.resize(root_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                        contours, _ = cv2.findContours(scaled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            cv2.drawContours(vis_img, contours, -1, (0, 0, 255), 2)
                except:
                    pass

            # Рисуем стебли
            for stem_mask in stem_masks:
                try:
                    if stem_mask.size > 0:
                        scaled_mask = cv2.resize(stem_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                        contours, _ = cv2.findContours(scaled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)
                except:
                    pass

            # Рисуем листья с номерами
            for j, leaf_mask in enumerate(leaf_masks):
                try:
                    if leaf_mask.size > 0:
                        scaled_mask = cv2.resize(leaf_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                        color = (255, 0, 0) if j % 2 == 0 else (255, 255, 0)
                        contours, _ = cv2.findContours(scaled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        if contours:
                            cv2.drawContours(vis_img, contours, -1, color, 2)

                            moments = cv2.moments(contours[0])
                            if moments["m00"] != 0:
                                cx = int(moments["m10"] / moments["m00"])
                                cy = int(moments["m01"] / moments["m00"])
                                cv2.putText(vis_img, str(j + 1), (cx, cy - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except:
                    continue

        measurements = {
            'image_name': Path(image_path).name,
            'plants': [plant_info],
            'total_plants': 1 if (root_masks or stem_masks or leaf_masks) else 0,
            'leaf_count': len(leaf_masks),
            'total_leaves_area_mm2': float(leaf_area_sum)
        }

        # сохраняем визуализацию
        if save_visualization and vis_img is not None:
            vis_path = Path('data/results/visualizations') / f"vis_{Path(image_path).name}"
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_path), vis_img)
            measurements['visualization_path'] = str(vis_path)

        return measurements
