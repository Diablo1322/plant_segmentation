import time
import json
import shutil
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from predict import PlantMeasurer


class NewImageHandler(FileSystemEventHandler):
    # обработчик новых файлов
    def __init__(self, image_watcher):
        self.image_watcher = image_watcher

    def on_created(self, event):
        # появился новый файл
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                time.sleep(1)
                self.image_watcher.process_file(file_path)


class ImageWatcher:
    def __init__(self,
                 input_folder="data/input",      # сюда кладут фото
                 output_folder="data/output",    # сюда ложатся результаты
                 model_path="models/best.pt",
                 calibration_file="calibration.json"):

        # создаём все папки проекта
        from utils import setup_project_structure
        setup_project_structure()

        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.model_path = model_path
        self.calibration_file = calibration_file

        # создаём папки если их нет
        self.input_folder.mkdir(parents=True, exist_ok=True)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # папка для обработанных фото
        self.processed_folder = self.input_folder / "done"
        self.processed_folder.mkdir(exist_ok=True)

        # загружаем модель
        print("Загружаю модель...")
        self.measurer = PlantMeasurer(model_path, calibration_file)
        print("Модель готова. Слежу за папкой:", self.input_folder)

        # чтобы не обрабатывать одно фото дважды
        self.processed_files = set()

        # для очистки старых файлов
        self.last_cleanup = time.time()

    def process_file(self, file_path):
        # обрабатывает одно изображение
        if file_path.name in self.processed_files:
            return

        print(f"Обработка: {file_path.name}")

        try:
            # запускаем модель
            result = self.measurer.process_image(str(file_path))

            # сохраняем визуализацию
            vis_name = f"vis_{file_path.stem}.jpg"
            vis_path = self.output_folder / vis_name
            if 'visualization_path' in result:
                shutil.copy(result['visualization_path'], vis_path)
                result['visualization_path'] = str(vis_path)

            # сохраняем JSON
            json_path = self.output_folder / f"{file_path.stem}.json"
            with open(json_path, 'w') as f:
                result_clean = result.copy()
                result_clean.pop('mask_path', None)
                json.dump(result_clean, f, indent=2)

            # добавляем в общий CSV
            self.update_csv(result, self.output_folder / "all_results.csv")

            # перемещаем исходное фото в обработанные
            file_path.rename(self.processed_folder / file_path.name)

            # запоминаем
            self.processed_files.add(file_path.name)

            print(f"Готово: {file_path.name} -> {json_path.name}")

        except Exception as e:
            print(f"Ошибка: {e}")
            # в папку с ошибками
            error_folder = self.input_folder / "errors"
            error_folder.mkdir(exist_ok=True)
            file_path.rename(error_folder / file_path.name)

    @staticmethod
    def update_csv(result, csv_path):
        # добавляет строку в CSV
        import pandas as pd

        new_row = {
            'image_name': result['image_name'],
            'root_length_mm': result['root']['length_mm'],
            'root_area_mm2': result['root']['area_mm2'],
            'stem_length_mm': result['stem']['length_mm'],
            'stem_area_mm2': result['stem']['area_mm2'],
            'leaf_count': result['leaf_count'],
            'total_leaves_area_mm2': result['total_leaves_area_mm2'],
            'processed_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        df.to_csv(csv_path, index=False)

    def cleanup_old_files(self, hours=3):
        # удаляет файлы старше N часов
        now = time.time()
        cutoff = now - (hours * 3600)

        # чистим папку с обработанными фото
        for f in self.processed_folder.glob("*"):
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                print(f"Удалено старое фото: {f.name}")

        # чистим папку с результатами (кроме all_results.csv)
        for f in self.output_folder.glob("*"):
            if f.is_file() and f.name != "all_results.csv" and f.stat().st_mtime < cutoff:
                f.unlink()
                print(f"Удален старый результат: {f.name}")

    def start(self):
        # запускает слежение
        event_handler = NewImageHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.input_folder), recursive=False)
        observer.start()

        try:
            while True:
                time.sleep(1)
                # проверяем каждые 30 минут
                if time.time() - self.last_cleanup > 1800:
                    self.cleanup_old_files(hours=3)
                    self.last_cleanup = time.time()
        except KeyboardInterrupt:
            observer.stop()
            print("\nОстановлено")

        observer.join()


if __name__ == "__main__":
    # запуск
    watcher = ImageWatcher(
        input_folder="data/input",
        output_folder="data/output",
        model_path="models/best.pt",
        calibration_file="calibration.json"
    )
    watcher.start()
