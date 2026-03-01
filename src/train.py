import torch
from ultralytics import YOLO


def train_segmentation_model(data_yaml_path, model_size='n'):
    # проверяем есть ли видеокарта NVIDIA, если нет - используем процессор
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используем устройство: {device}")

    # загружаем предобученную модель
    model = YOLO(f'yolov8{model_size}-seg.pt')

    # запускаем обучение
    model.train(
        data=data_yaml_path,  # файл с путями к размеченным данным
        epochs=100,  # сколько раз пройдем по всем картинкам
        imgsz=320,  # размер картинок, подаваемых в сеть
        batch=1,  # сколько картинок обрабатывать за раз
        device=device,
        patience=20,  # если 20 эпох нет улучшений - останавливаем
        save=True,  # сохранять лучшую модель
        project='plant_segmentation',  # папка для сохранения
        name='wheat_rocket',  # имя эксперимента
        exist_ok=True,  # перезаписывать если папка уже есть
        pretrained=True,  # использовать предобученные веса
        optimizer='AdamW',  # оптимизатор (метод обучения)
        lr0=0.001,  # начальная скорость обучения
        augment=True,  # применять аугментации
        seed=42  # для воспроизводимости результатов
    )

    return model


if __name__ == "__main__":
    # путь к data.yaml
    data_yaml = "data/raw/data.yaml"
    model = train_segmentation_model(data_yaml, model_size='s')
