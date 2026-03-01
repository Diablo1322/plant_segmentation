from pathlib import Path
from datetime import datetime


def setup_project_structure():
    # Создаёт все нужные папки при первом запуске
    directories = [
        'data/raw',  # исходные фото для обучения
        'data/calibrated',  # фото шахматки
        'data/input',  # сюда бот/сайт кидают фото
        'data/input/done',  # обработанные фото
        'data/input/errors',  # фото с ошибками
        'data/output',  # результаты (JSON, визуализации, CSV)
        'models',  # сохранённые модели
        'logs',  # логи работы
        'src'  # код (если вдруг нет)
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("Структура папок создана")


def get_image_paths(folder, extensions=None):
    # Возвращает список всех картинок в папке
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png']

    folder = Path(folder)
    images = []
    for ext in extensions:
        # ищем с маленькими и большими расширениями
        images.extend(folder.glob(f'*{ext}'))
        images.extend(folder.glob(f'*{ext.upper()}'))
    return sorted(images)


def log_processing(image_name, status, details=""):
    # Записывает в лог информацию об обработке
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {image_name}: {status} {details}"

    log_file = Path('logs/processing.log')
    log_file.parent.mkdir(exist_ok=True)

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')

    print(log_entry)


def log_error(image_name, error):
    # Записывает ошибку в лог
    log_processing(image_name, "ERROR", str(error))


def validate_calibration(calibration_data):
    # Проверяет что калибровка прошла успешно
    if 'pixels_per_mm' not in calibration_data:
        return False, "Нет данных о масштабе"
    if calibration_data['pixels_per_mm'] <= 0:
        return False, "Масштаб должен быть положительным числом"
    return True, "OK"
