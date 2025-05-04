HELP_MESSAGES = {
    "version": {
        "ru": "Показать версию программы.",
        "en": "Show program version."
    },
    "lang": {
        "ru": "Смена языка интерфейса программы.",
        "en": "Change program interface language."
    },
    "main_description": {
        "ru": "Программа запущена без команд и флагов!",
        "en": "Program started without commands or flags!"
    },
    "models": {
        "ru": "Работа с классификационными моделями.",
        "en": "Work with classification models."
    },
    "config": {
        "ru": "Загрузка конфиг файла.",
        "en": "Load configuration file."
    },
    "config_path": {
        "ru": "Путь к конфигурационному файлу.",
        "en": "Path to configuration file."
    },
    "install": {
        "ru": "Установка моделей из онлайна.",
        "en": "Install models from online source."
    },
    "list": {
        "ru": "Вывод списка моделей.",
        "en": "List available models."
    },
    "list_online": {
        "ru": "Вывод моделей, доступных онлайн.",
        "en": "List models available online."
    },
    "predict": {
        "ru": "Сделать предсказание.",
        "en": "Make a prediction."
    },
    "input_file": {
        "ru": "Путь к входному файлу.",
        "en": "Path to input file."
    },
    "output_file": {
        "ru": "Путь к результирующему файлу.",
        "en": "Path to output file."
    },
    "input_encode": {
        "ru": "Входная кодировка.",
        "en": "Input file encoding."
    },
    "output_encode": {
        "ru": "Выходная кодировка.",
        "en": "Output file encoding."
    },
    "identifier": {
        "ru": "Идентификатор рубрикатора.",
        "en": "Rubricator identifier."
    },
    "packet": {
        "ru": "Размер пакета текстов для классификации.",
        "en": "Batch size for classification."
    },
    "input_format": {
        "ru": "Формат файла.",
        "en": "Input file format."
    },
    "language": {
        "ru": "Язык.",
        "en": "Language."
    },
    "threshold": {
        "ru": "Минимальная вероятность рубрики (порог).",
        "en": "Minimum rubric probability (threshold)."
    },
    "normalization": {
        "ru": "Нормализация.",
        "en": "Normalization."
    },
    "device": {
        "ru": "Выбор устройства выполнения.",
        "en": "Choose computation device."
    },
    "dialog": {
        "ru": "Диалоговое окно.",
        "en": "Dialog window."
    },
    "silence": {
        "ru": "Включение режима без логов.",
        "en": "Enable silent mode (no logs)."
    },
    "workers": {
        "ru": "Максимальное количество процессов для загрузки данных.",
        "en": "Maximum number of data loading processes."
    },
    "server": {
        "ru": "Запустить сервер.",
        "en": "Start server."
    },
    "host": {
        "ru": "IP адрес.",
        "en": "IP address."
    },
    "port": {
        "ru": "Порт.",
        "en": "Port."
    },
    "device": {
        "ru": "Выбор устройства выполнения.",
        "en": "Selecting the execution device."
    },
    "model": {
        "ru": "Модель для установки.",
        "en": "Model for downloading."
    },
    "help": {
        "ru": "Отображение доступных флагов и функций.",
        "en": "Display available flags and functions."
    }
}

import json

try:
    with open("prog.json", "r", encoding="cp1251") as file:
        prog = json.load(file)
        lang = prog["language"]

except:
    print("Can't open prog.json")
    lang = "ru"


def get_help(key: str) -> str:
    return HELP_MESSAGES.get(key, {}).get(lang, "")