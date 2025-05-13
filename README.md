# VINITI_Text_Classification

## Датасеты
Ввиду ограничений на размер файлов в github, необходимо скачать папку [datasets](https://vvzunin.me:10003/d/s/xfHnY40x4YOicNSECWYRfpbL6wxkXcZS/1ZHuNDJrT7zJV-l3JLp_FWtvbrUlfMPn-rrMgTa1ZSws) для дальнейшей работы в данную директорию.

## Веса
Все полученные веса для различных нейронных сетей находятся [здесь](https://vvzunin.me:10003/d/s/y7O7ow4NqStgYMnktinhk5Tvr9jAneYR/cZ9tgTKahzpz0Qs1Bxs0Iyby8DBB6sS6-QLygJNtVSws).

## Графический интерфейс

Веб-приложение для автоматической классификации текстовых документов по рубрикам ГРНТИ с использованием моделей глубокого обучения.

## Требования

- Python 3.8+
- CUDA (рекомендуется для GPU-ускорения)

## Структура проекта
```
└── 📁src
    └── 📁backend                            # Папка для backend
        └── 📁app
            └── __init__.py                   # Папка для backend
            └── api.py                        # Код для релизации backend
            └── prediction.py                 # Код для получения рубрик ГРНТИ
        └── 📁bert_peft_level1_extra             # Модель для классификации 1-го уровня ГРНТИ
            └── my_grnti1_int.json  # Словарь кодировок рубрик для предсказания 1-го уронвя ГРНТИ
        └── 📁bert_peft_level2_with_labels_extra # Модель для классификации 2-го уровня ГРНТИ
            └── my_grnti2_int.json  # Словарь кодировок рубрик для предсказания 2-го уронвя ГРНТИ
        └── GRNTI_1_ru.json                   # Словарь названий рубрик для 1-го уровня ГРНТИ
        └── GRNTI_2_ru.json                   # Словарь названий рубрик для 2-го уровня ГРНТИ
        └── GRNTI_3_ru.json                   # Словарь названий рубрик для 3-го уровня ГРНТИ
        └── main.py                           # Программа запуска сервера для backend
    └── 📁frontend                            # Папка для frontend
        └── 📁css                
            └── styles.css                     # Файл для стилей
        └── index.html                         # html страница приложения
        └── 📁js
            └── api.js                         # Взаимодействие frontend и backend
            └── fileHandler.js                 # Настройка процесса получения файлов
            └── main.js                        # Основная логика для графического интерфеса
            └── config_backend.js              # Получение port и host backend
    └── 📁static
        └── config.json                        # Файл json с port и host для backend
└── 📁train
    └── preprocessing.py # Модуль предобработки данных 
    └── test_prediction.py # Модуль создания статистических отчетов по результатам классификации
    └── train.py # Модуль обучения
    └── TrainSettings.py # Параметры обучения
└── 📁examples # Примеры работы CLI
└── 📁docs # Документация CLI
└── 📁dicts # Словари с кодировками рубрик ГРНТИ
└── 📁toEXE # Функционал для упаковки создания .exe файла для (CLI)
└── config.json            # Основной конфигурационный файл (CLI)
└── config.py              # Модуль для работы с конфигурацией (CLI)
└── Dockerfile             # Файл для создания Docker-образа backend
└── help_message.py        # Сообщения справки (CLI)
└── main.py                # Главный модуль приложения (CLI)
└── messages.py            # Сообщения для пользовательского интерфейса 
└── prediction.py          # Логика предсказаний и классификации (CLI)
└── prog.json              # Настройки программы (CLI)
└── requirements.txt       # Зависимости для локальной установки (CLI)
└── requirements_docker.txt # Зависимости для Docker-контейнера (backend)
└── run-container.sh       # Скрипт для запуска Docker-контейнера (backend)
```
## API Endpoints
POST /classify - Основной endpoint для классификации текстов
Параметры:

files: Список текстовых файлов

level1: Включить классификацию 1-го уровня (bool)

level2: Включить классификацию 2-го уровня (bool)

level3: Включить классификацию 3-го уровня (bool)

decoding: Включить расшифровку кодов (bool)

threshold: Порог вероятности (float)