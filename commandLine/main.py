description = {
  "ru": "Автоматизированный классификатор текстов по кодам ГРНТИ.",
  "en": "Automation text classifier for GRNTI codes."
}

arguments = {
  "i": {
    "name": "-i",
    "default": "text.txt",
    "type": str,
    "choices": None,
    "required": True,
    "help": {
      "ru": "Путь к существующему файлу (по-умолчанию: %(default)s)",
      "en": "Path to an existing file (default: %(default)s)"
    },
    "metavar": "input_file.txt",
    "dest": "inFile"
  },
  "o": {
    "name": "-o",
    "default": "results.csv",
    "type": str,
    "choices": None,
    "required": True,
    "help": {
      "ru": "Путь к результирующему файлу (по-умолчанию: %(default)s)",
      "en": "Path to an resulting file (default: %(default)s)"
    },
    "metavar": "output_file.csv",
    "dest": "outFile"
  },
  "id": {
    "name": "-id",
    "default": "RGNTI3",
    "type": str,
    "choices": ["RGNTI1", "RGNTI2", "RGNTI3"],
    "required": True,
    "help": {
      "ru": "Идентификатор рубрикатора (по-умолчанию: %(default)s)",
      "en": "Rubricator ID (default: %(default)s)"
    },
    "metavar": "RGNTI(/1/2/3)",
    "dest": "level"
  },
  "f": {
    "name": "-f",
    "default": "plain",
    "type": str,
    "choices": ["plain", "multidoc"],
    "required": True,
    "help": {
      "ru": "Формат файла (по-умолчанию: %(default)s)",
      "en": "File format (default: %(default)s)"
    },
    "metavar": "format",
    "dest": "format"
  },
  "l": {
    "name": "-l",
    "default": "ru",
    "type": str,
    "choices": ["ru", "en"],
    "required": True,
    "help": {
      "ru": "Язык текстов (по-умолчанию: %(default)s)",
      "en": "Text language (default: %(default)s)"
    },
    "metavar": "language",
    "dest": "language"
  },
  "t": {
    "name": "-t",
    "default": 0.5,
    "type": float,
    "choices": None,
    "required": True,
    "help": {
      "ru": "Минимальная вероятность рубрики (по-умолчанию: %(default)s)",
      "en": "Minimum rubric probability (default: %(default)s)"
    },
    "metavar": "threshold",
    "dest": "threshold"
  },
  "n": {
    "name": "-n",
    "default": "not",
    "type": str,
    "choices": ["not", "some", "all"],
    "required": False,
    "help": {
      "ru": "Нормализация результатов (по-умолчанию: %(default)s)",
      "en": "Results normalisation (default: %(default)s)"
    },
    "metavar": "normalisation",
    "dest": "normalisation"
  },
}

lang = "ru"
datetimeFormatOutput = "%d.%m.%Y %H:%M:%S.%f"
models = {
  "lora": {
    1: "..\\models\\bert\\expriment_save_model",
    2: "..\\models\\bert\\expriment_save_model2",
    3: ""
  }
}

if __name__ == "__main__":
  import datetime
  start = datetime.datetime.now()
  if (lang == "ru"):
    print("{} Программа запущена.".format(start.strftime(datetimeFormatOutput)))
  elif (lang == "en"):
    print("{} Programm started.".format(start.strftime(datetimeFormatOutput)))

  import argparse
  from prediction import prepair_model, prepair_data_level1, prepair_data_level2,\
  prepair_dataset, make_predictions, save_rubrics_names
  import torch

  libs = datetime.datetime.now()
  if (lang == "ru"):
    print("{} Библиотеки импортированы.".format(libs.strftime(datetimeFormatOutput)))
  elif (lang == "en"):
    print("{} Libraries imported.".format(libs.strftime(datetimeFormatOutput)))

  parser = argparse.ArgumentParser(description=description[lang])
  for i in arguments:
    parser.add_argument(
      arguments[i]["name"],
      default=arguments[i]["default"],
      type=arguments[i]["type"],
      choices=arguments[i]["choices"],
      required=arguments[i]["required"],
      help=arguments[i]["help"][lang],
      metavar=arguments[i]["metavar"],
      dest=arguments[i]["dest"]
      )
    
  args = parser.parse_args()
  torch.cuda.empty_cache()

  model1 = None if models ["lora"][1] == "" else prepair_model(n_classes=31, lora_model_path=models["lora"][1])
  model2 = None
  model3 = None
  print("-1-")
  if ((args.level == "RGNTI2") or (args.level == "RGNTI3")):
    model2 = None if models ["lora"][2] == "" else prepair_model(n_classes=246, lora_model_path=models["lora"][2])
    print("-2-")
  if (args.level == "RGNTI3"):
    model3 = None if models ["lora"][3] == "" else prepair_model(n_classes=0, lora_model_path=models["lora"][3])
    print("-3-")
  
  if ((model1 is None) or
      ((model2 is None) and ((args.level == "RGNTI2") or (args.level == "RGNTI3"))) or
      ((model3 is None) and (args.level == "RGNTI3"))):
    if (lang == "ru"):
      print("{} Одна из необходимых моделей для вычислений не загружена или не найдена."
            .format(libs.strftime(datetimeFormatOutput)))
    elif (lang == "en"):
      print("{} One of the models required for calculations was not loaded or not found."
            .format(libs.strftime(datetimeFormatOutput)))
    exit()

  modelsTime = datetime.datetime.now()
  if (lang == "ru"):
    print("{} Модели загружены.".format(modelsTime.strftime(datetimeFormatOutput)))
  elif (lang == "en"):
    print("{} Models loaded.".format(modelsTime.strftime(datetimeFormatOutput)))


  df_test = prepair_data_level1(args.inFile, format=args.format)
  dataset_test = prepair_dataset(df_test)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  deviceTime = datetime.datetime.now()
  if (lang == "ru"):
    print("{} Устройство выполнения: {}".format(deviceTime.strftime(datetimeFormatOutput), device))
  elif (lang == "en"):
    print("{} Device: {}".format(deviceTime.strftime(datetimeFormatOutput), device))

  predictions_level1 = make_predictions(model1, dataset_test, device=device, threshold=0.5)
  save_rubrics_names(predictions_level1, path_to_csv = "result1.csv")


  del model1
  torch.cuda.empty_cache()
  print("Part for second level")
  

  df_test2 = prepair_data_level2(df_test, predictions_level1)

  dataset_test2 = prepair_dataset(df_test2)

  predictions_level2 = make_predictions(model2, dataset_test2, device=device, threshold=0.5)

  save_rubrics_names(predictions_level2, path_to_csv = "result2.csv")

  finish = datetime.datetime.now()
  if (lang == "ru"):
    print("{} Программа завершена.".format(finish.strftime(datetimeFormatOutput)))
  elif (lang == "en"):
    print("{} Programm finished.".format(finish.strftime(datetimeFormatOutput)))