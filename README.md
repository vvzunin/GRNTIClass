# GRNTIClass - классфикация научных текстов по кодам ГРНТИ

## Датасеты
Ввиду ограничений на размер файлов в github, необходимо скачать папку [datasets](https://vvzunin.me:10003/d/s/xfHnY40x4YOicNSECWYRfpbL6wxkXcZS/1ZHuNDJrT7zJV-l3JLp_FWtvbrUlfMPn-rrMgTa1ZSws) для дальнейшей работы в данную директорию.

## Веса
Все полученные веса для различных нейронных сетей находятся [здесь](https://vvzunin.me:10003/d/s/y7O7ow4NqStgYMnktinhk5Tvr9jAneYR/cZ9tgTKahzpz0Qs1Bxs0Iyby8DBB6sS6-QLygJNtVSws).

## Графический интерфейс

Веб-приложение для автоматической классификации текстовых документов по рубрикам ГРНТИ с использованием моделей глубокого обучения.

## Руководство пользователя

Руководство пользователя доступно [здесь](doc/manual.md).

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

## Дополнительная информация

Больше информации о предыдущих версиях и разработках можно найти в следующих публикация и репозиториях:
1. Романов, А. Ю. Применение методов машинного обучения для решения задачи автоматической рубрикации статей по УДК / А. Ю. Романов, К. Е. Ломотин, Е. С. Козлова // Информационные технологии. – 2017. – Т. 23, № 6. – С. 418-423. – EDN YSLQPP.
2. Romanov, A., Kozlova, E., Lomotin, K. (2018). Application of NLP Algorithms: Automatic Text Classifier Tool. In: Alexandrov, D., Boukhanovsky, A., Chugunov, A., Kabanov, Y., Koltsova, O. (eds) Digital Transformation and Global Society. DTGS 2018. Communications in Computer and Information Science, vol 859. Springer, Cham. doi: 10.1007/978-3-030-02846-6_25
3. A. Romanov, K. Lomotinand E. Kozlova, “Application of Natural Language Processing Algorithms to the Task of Automatic Classification of Russian Scientific Texts”, <i>Data Science Journal</i>, vol. 18, no. 1, p. 37, 2019, doi: 10.5334/dsj-2019-037.
4. Kusakin, I.K., Fedorets, O.V. & Romanov, A.Y. Classification of Short Scientific Texts. Sci. Tech. Inf. Proc. 50, 176–183 (2023). doi: 10.3103/S0147688223030024
5. ATC_Automatic-Text-Classification // GitHub URL: https://github.com/RomeoMe5/ATC_Automatic-Text-Classification
6. ECS_Experiment-Conduction-System // GitHub URL: https://github.com/RomeoMe5/ECS_Experiment-Conduction-System
