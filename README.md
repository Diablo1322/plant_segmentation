# plant_segmentation
Plant Segmentation - README
Проект для автоматической сегментации и измерения корня, стебля и листьев растений (пшеница и руккола).
Модель определяет каждый лист отдельно, считает длину корня/стебля и площадь всех частей.

📁 Структура репозитория
plant_segmentation/

│

├── src/

│   ├── calibrate.py      # калибровка камеры по шахматке

│   ├── train.py          # обучение модели

│   ├── predict.py        # класс для измерений

│   ├── utils.py          # вспомогательные функции

│   └── watcher.py        # автоматическая обработка новых фото

│

├── models/               # сюда сохраняются обученные модели

├── requirements.txt      # зависимости

└── README.md

🚀 Быстрый старт
1. Установка
# клонируем репозиторий
git clone https://github.com/your-repo/plant_segmentation.git
cd plant_segmentation

# создаём виртуальное окружение
python -m venv venv

# активируем
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# ставим зависимости
pip install -r requirements.txt
2. Подготовка данных
Создай вручную папку data и внутри неё:

data/

├── raw/           # сюда кладёшь все фото для обучения (пшеница + руккола)

├── calibrated/    # сюда кладёшь фото шахматки (calib_1.jpg ... calib_18.jpg)

├── input/         # сюда будут кидать фото для обработки (создастся автоматом)

└── output/        # сюда будут ложиться результаты (создастся автоматом)

3. Калибровка камеры
python src/calibrate.py
Программа найдёт все фото шахматки, усреднит результаты и создаст файл calibration.json в корне проекта.

🎯 Разметка данных (Roboflow)
1. Создание проекта
Зайди на Roboflow

Нажми "Create New Project"

Название: plant-segmentation

Project Type: Instance Segmentation

Жми "Create Project"

2. Загрузка фото
Нажми "Upload" или перетащи все фото из папки data/raw/

После загрузки нажми "Save and Continue"

3. Разметка классов
Создай три класса:

root - корень

stem - стебель

leaf - лист

4. Разметка изображений
Открываешь фото

Инструмент Polygon (или Smart Polygon)

Важно: каждый лист обводи отдельным полигоном!

После каждого фото жми Save

Разметь минимум 50-100 фото (чем больше, тем лучше).

5. Экспорт датасета
В левом меню выбери "Versions" → "Create New Version"

Добавь аугментации (повороты, яркость, шум)

Жми "Generate"

После генерации нажми "Export Dataset"

Выбери формат YOLO v8

Скачай ZIP и распакуй в папку roboflow/ в корне проекта

Структура после распаковки:

roboflow/

├── train/

├── valid/

├── test/

└── data.yaml

🧠 Обучение модели
1. Проверь видеокарту (опционально)
python -c "import torch; print(torch.cuda.is_available())"
Если False - будет учиться на процессоре (очень долго).

2. Запусти обучение
python src/train.py
Параметры (можно менять в train.py):

model_size='s' - размер модели ('n' - nano, 's' - small, 'm' - medium)

epochs=100 - количество эпох

batch=16 - размер пакета (уменьши если мало памяти)

imgsz=640 - размер картинки (уменьши до 320 если мало памяти)

3. Где результат
После обучения модель сохранится в:

runs/segment/plant_segmentation/wheat_rocket/weights/best.pt
Скопируй её в папку models/:

copy runs\segment\plant_segmentation\wheat_rocket\weights\best.pt models\
🤖 Автоматическая обработка
1. Запусти watcher
python src/watcher.py
Программа будет следить за папкой data/input/.

2. Как пользоваться
Кидаешь любое фото в data/input/

Через несколько секунд в data/output/ появляются:

vis_photo.jpg - фото с нарисованными контурами (готово для бота/сайта)

photo.json - результаты измерений

all_results.csv - общая таблица (обновляется)

3. Структура папок после обработки
data/

├── input/

│   ├── done/       # обработанные фото (перемещаются сюда)

│   └── errors/     # фото с ошибками

└── output/

    ├── vis_*.jpg   # визуализации
    
    ├── *.json      # результаты по каждому фото
    
    └── all_results.csv
    
📊 Формат результатов
JSON для одного фото:
{
  "image_name": "wheat_001.jpg",
  "root": {
    "length_mm": 45.23,
    "area_mm2": 12.45
  },
  "stem": {
    "length_mm": 23.67,
    "area_mm2": 8.91
  },
  "leaves": [
    {"leaf_id": 1, "area_mm2": 32.45},
    {"leaf_id": 2, "area_mm2": 28.91}
  ],
  "leaf_count": 5,
  "total_leaves_area_mm2": 156.34,
  "visualization_path": "data/output/vis_wheat_001.jpg"
}
CSV (все результаты):
image_name	root_length_mm	root_area_mm2	stem_length_mm	stem_area_mm2	leaf_count	total_leaves_area_mm2
wheat_001.jpg	45.23	12.45	23.67	8.91	5	156.34
⚙️ Настройки
Если мало видеопамяти
В train.py уменьши:

imgsz=320,     # вместо 640
batch=4,       # вместо 16
Если модель ничего не находит
Добавь ещё размеченных фото

Проверь качество разметки в Roboflow

Увеличь количество эпох: epochs=200

Очистка старых файлов
Watcher автоматически удаляет файлы старше 24 часов из input/done/ и output/ (кроме all_results.csv).

🔄 Интеграция с ботом/сайтом
Твоя часть готова. Остальным нужно:

Класть фото в data/input/

Забирать результаты из data/output/

vis_*.jpg - фото с контурами

*.json - данные по конкретному фото

all_results.csv - общая таблица

🐛 Возможные проблемы
Ошибка: "No module named 'cv2'"

pip install opencv-python
Ошибка: "CUDA out of memory"
Уменьши batch и imgsz в train.py

Шахматка не находится
Проверь размер в calibrate.py: chessboard_size=(7, 4) для 5x8 клеток

Модель ничего не находит
Мало данных или плохая разметка. Добавь ещё фото в Roboflow.
