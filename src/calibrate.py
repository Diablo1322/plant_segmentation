import cv2
import numpy as np
import json
import glob


class CameraCalibrator:
    def __init__(self, chessboard_size=(9, 6), square_size_mm=10):
        # размер шахматки
        self.chessboard_size = chessboard_size
        self.chessboard_size = chessboard_size
        self.square_size_mm = square_size_mm  # размер клетки в мм

    def calibrate_from_image(self, image_path):
        # загружаем фото шахматки
        img = cv2.imread(str(image_path))
        if img is None:
            return {'success': False, 'error': 'Не удалось загрузить изображение'}

        # переводим в ч/б для поиска углов
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ищем углы шахматной доски
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

        if ret:
            # уточняем положение углов для большей точности
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # берем первый и последний угол
            first_corner = corners_refined[0][0]
            last_corner = corners_refined[-1][0]

            # расстояние в пикселях между ними
            distance_pixels = np.linalg.norm(last_corner - first_corner)

            # реальное расстояние в мм
            squares_x = self.chessboard_size[0] - 1
            squares_y = self.chessboard_size[1] - 1
            real_distance_mm = np.sqrt((squares_x * self.square_size_mm) ** 2 +
                                       (squares_y * self.square_size_mm) ** 2)

            # сколько пикселей в 1 мм
            pixels_per_mm = distance_pixels / real_distance_mm

            return {
                'success': True,
                'pixels_per_mm': pixels_per_mm,
                'image_shape': img.shape[:2]
            }
        else:
            return {'success': False, 'error': 'Шахматка не найдена на изображении'}

    def calibrate_from_multiple(self, image_pattern="data/calibrated/calib_*.jpg"):
        # калибровка по нескольким фото для большей точности
        images = glob.glob(image_pattern)
        if not images:
            return {'success': False, 'error': 'Нет файлов для калибровки'}

        print(f"Найдено {len(images)} фото для калибровки")

        pixels_per_mm_list = []
        success_count = 0

        for img_path in images:
            result_calib = self.calibrate_from_image(img_path)
            if result_calib['success']:
                pixels_per_mm_list.append(result_calib['pixels_per_mm'])
                success_count += 1

        if success_count == 0:
            return {'success': False, 'error': 'Ни на одном фото не найдена шахматка'}

        # усредняем результаты
        avg_pixels_per_mm = np.mean(pixels_per_mm_list)

        return {
            'success': True,
            'pixels_per_mm': avg_pixels_per_mm,
            'used_images': success_count,
            'total_images': len(images)
        }

    @staticmethod
    def save_calibration(calibration_data, save_path='calibration.json'):
        # сохраняем результаты в файл
        with open(save_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"Калибровка сохранена в {save_path}")


if __name__ == "__main__":
    # запуск калибровки по всем 18 фото
    cal = CameraCalibrator(chessboard_size=(7, 4), square_size_mm=10)  # размер из твоего скрина
    result = cal.calibrate_from_multiple("data/calibrated/calib_*.jpg")

    if result['success']:
        cal.save_calibration(result, "calibration.json")
        print(f"Калибровка по {result['used_images']} фото")
        print(f"Средний масштаб: {result['pixels_per_mm']:.2f} px/мм")
    else:
        print("Ошибка:", result['error'])
