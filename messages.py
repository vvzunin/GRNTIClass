import datetime

datetimeFormatOutput = "%d.%m.%Y %H:%M:%S.%f"

messagesTexts = {
  "start": {
    "ru": "%s Программа запущена.",
    "en": "%s Programm started."
  },
  "finish": {
    "ru": "%s Программа завершена.",
    "en": "%s Programm finished."
  },
  "libs": {
    "ru": "%s Библиотеки импортированы.",
    "en": "%s Libraries imported."
  },
  "modelError": {
    "ru": "%s Одна из необходимых моделей для вычислений не загружена или не найдена.",
    "en": "%s One of the models required for calculations was not loaded or not found."
  },
  "modelLoaded": {
    "ru": "%s Модели загружены.",
    "en": "%s Models loaded."
  },
  "device": {
    "ru": "%s Устройство выполнения: %s",
    "en": "%s Device: %s"
  },
  "startPredict": {
    "ru": "%s Начинается предсказание результатов",
    "en": "%s The prediction of the results begins"
  },
  "notComplete": {
    "ru": "%s Данный функционал еще не реализован",
    "en": "%s This functionality is still"
  },
  "badFlag": {
    "ru": "%s Флаг %s не поддерживает значение %s",
    "en": "%s Flag %s is not supported value %s"
  },
  "badConfig": {
    "ru": "%s Конфиг файл не найден или недоступен",
    "en": "%s Config file not found or unavailable"
  },
  "backendError": {
    "ru": "%s Не удаось запустить backend",
    "en": "%s Couldn't deploy the backend"
  }
}

def printMessage(messageType, lang = "ru", args = ()):
  args = (datetime.datetime.now().strftime(datetimeFormatOutput), ) + args
  print(messagesTexts[messageType][lang] % args)