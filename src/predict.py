from ultralytics import YOLO
import cv2
import numpy as np
import json
from pathlib import Path
import pandas as pd
from skimage import morphology
from utils import (
    setup_project_structure,
    get_image_paths,
    log_processing,
    log_error
)


class PlantMeasurer:
    def __init__(self, model_path, calibration_file):
        # загружаем обученную модель
        self.model = YOLO(model_path)

        # загружаем данные калибровки
        with open(calibration_file, 'r') as f:
            self.calibration = json.load(f)

        self.pixels_per_mm = self.calibration['pixels_per_mm']
        self.mm_per_pixel = 1.0 / self.pixels_per_mm
        self.area_per_pixel_mm2 = self.mm_per_pixel ** 2  # площадь одного пикселя в мм²

        print(f"Масштаб: {self.pixels_per_mm:.2f} px/мм")
        print(f"Площадь пикселя: {self.area_per_pixel_mm2:.4f} мм²")

        # создаем нужные папки
        setup_project_structure()

    def measure_length(self, mask):
        # скелетизация - находим центральную линию объекта
        skeleton = morphology.skeletonize(mask)
        length_pixels = np.sum(skeleton)  # считаем пиксели в скелете
        length_mm = length_pixels * self.mm_per_pixel  # переводим в миллиметры
        return length_mm

    def measure_area(self, mask):
        # считаем площадь в пикселях и переводим в мм
        area_pixels = np.sum(mask > 0)
        area_mm2 = area_pixels * self.area_per_pixel_mm2
        return area_mm2

    def process_image(self, image_path, save_visualization=True):
        # Обрабатывает одно изображение
        # загружаем изображение
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb)[0]

        measurements = {
            'image_name': Path(image_path).name,
            'root': {'length_mm': 0.0, 'area_mm2': 0.0},
            'stem': {'length_mm': 0.0, 'area_mm2': 0.0},
            'leaves': []
        }

        # создаём копию для визуализации
        vis_img = None
        if save_visualization:
            vis_img = img.copy()

        leaf_count = 0

        for mask, cls_id in zip(results.masks.data, results.boxes.cls):
            cls_name = results.names[int(cls_id.item())]
            mask_np = mask.cpu().numpy().astype(np.uint8)
            area_mm2 = self.measure_area(mask_np)

            if cls_name == 'root':
                length_mm = self.measure_length(mask_np)
                measurements['root']['length_mm'] = float(length_mm)
                measurements['root']['area_mm2'] = float(area_mm2)

                if save_visualization and vis_img is not None:
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_img, contours, -1, (0, 0, 255), 2)

            elif cls_name == 'stem':
                length_mm = self.measure_length(mask_np)
                measurements['stem']['length_mm'] = float(length_mm)
                measurements['stem']['area_mm2'] = float(area_mm2)

                if save_visualization and vis_img is not None:
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)

            elif cls_name == 'leaf':
                leaf_count += 1
                leaf_data = {
                    'leaf_id': leaf_count,
                    'area_mm2': float(area_mm2)
                }
                measurements['leaves'].append(leaf_data)

                if save_visualization and vis_img is not None:
                    # чередуем цвета для разных листьев
                    color = (255, 0, 0) if leaf_count % 2 == 0 else (255, 255, 0)
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_img, contours, -1, color, 2)

                    # подписываем номер листа
                    if contours and len(contours) > 0:
                        n_leaf = cv2.moments(contours[0])
                        if n_leaf["m00"] != 0:
                            cx = int(n_leaf["m10"] / n_leaf["m00"])
                            cy = int(n_leaf["m01"] / n_leaf["m00"])
                            cv2.putText(vis_img, str(leaf_count), (cx, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        measurements['total_leaves_area_mm2'] = float(sum(leaf['area_mm2'] for leaf in measurements['leaves']))
        measurements['leaf_count'] = leaf_count

        # сохраняем визуализацию
        if save_visualization and vis_img is not None:
            # папка для визуализаций
            vis_path = Path('data/results/visualizations') / f"vis_{Path(image_path).name}"
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_path), vis_img)
            measurements['visualization_path'] = str(vis_path)

        return measurements

    def process_batch(self, image_folder, output_csv='measurements.csv'):
        # получаем список всех изображений
        image_paths = get_image_paths(image_folder)

        all_results = []

        for i, img_path in enumerate(image_paths):
            print(f"Обработка {i + 1}/{len(image_paths)}: {img_path.name}")
            log_processing(img_path.name, "START")

            try:
                results = self.process_image(img_path)
                all_results.append(results)
                log_processing(img_path.name, "SUCCESS")

                # сохраняем промежуточные результаты каждые 10 штук
                if (i + 1) % 10 == 0:
                    self.save_results(all_results, f'partial_results_{i + 1}.json')

            except Exception as e:
                print(f"Ошибка: {e}")
                log_error(img_path.name, e)
                continue

        # сохраняем все результаты
        self.save_results(all_results, output_csv)
        return all_results

    @staticmethod
    def save_results(results, output_path):
        # готовим данные для таблицы
        flat_results = []

        for res in results:
            flat_row = {
                'image_name': res['image_name'],
                'root_length_mm': res['root']['length_mm'],
                'root_area_mm2': res['root']['area_mm2'],
                'stem_length_mm': res['stem']['length_mm'],
                'stem_area_mm2': res['stem']['area_mm2'],
                'leaf_count': res['leaf_count'],
                'total_leaves_area_mm2': res['total_leaves_area_mm2']
            }
            flat_results.append(flat_row)

        # сохраняем в CSV
        df = pd.DataFrame(flat_results)
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        else:
            # если расширение не csv, сохраняем в JSON
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

        print(f"Результаты сохранены в {output_path}")

        # выводим статистику
        print("\nСТАТИСТИКА:")
        print(f"Всего обработано: {len(results)} изображений")
        print(f"Средняя длина корня: {df['root_length_mm'].mean():.2f} мм")
        print(f"Средняя длина стебля: {df['stem_length_mm'].mean():.2f} мм")
        print(f"Среднее количество листьев: {df['leaf_count'].mean():.1f}")
        print(f"Средняя площадь листьев: {df['total_leaves_area_mm2'].mean():.2f} мм²")
