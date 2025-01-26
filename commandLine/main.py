prog = {
  'name': 'GRNTIClass',
  'version': '1.1.0'
}

description = {
  "ru": "Автоматизированный классификатор текстов по кодам ГРНТИ. {} {}".format(prog['name'], prog['version']),
  "en": "Automation text classifier for GRNTI codes. {} {}".format(prog['name'], prog['version']),
}
mesages = {
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
  }
}

import os

models = {
  "lora": {
    1: os.path.dirname(os.path.abspath(__file__)) + "\\..\\models\\bert2\\bert_peft_level1",
    2: os.path.dirname(os.path.abspath(__file__)) + "\\..\\models\\bert2\\bert_peft_level2_with_labels",
    3: ""
  }
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
  "m": {
    "name": "-m",
    "default": list(models.keys())[0],
    "type": str,
    "choices": list(models.keys()),
    "required": False,
    "help": {
      "ru": "Модель для классификации (по-умолчанию: %(default)s)",
      "en": "Model for classification (default: %(default)s)"
    },
    "metavar": "modelType",
    "dest": "modelType"
  },
  "d": {
    "name": "-d",
    "default": False,
    "type": int,
    "choices": None,
    "required": False,
    "help": {
      "ru": "Использованием диалогового взаимодействия с программой",
      "en": "Using dialogue interaction with the program"
    },
    "metavar": "dialogue",
    "dest": "dialogue"
  },
}

lang = "ru"
datetimeFormatOutput = "%d.%m.%Y %H:%M:%S.%f"

def printInfo(formatString, args = []):
  print(formatString[lang] % args)

def get_user_inputs():
    user_args = {}
    args = arguments.copy()
    del args['d']
    for arg_key, arg_info in args.items():
        prompt = (f"{arg_info['help'][lang]}: ") % {'default': arg_info['default']}
        user_input = input(prompt)
        if user_input == "":
            user_input = arg_info['default']
        user_args[arg_info['dest']] = arg_info['type'](user_input)
    return user_args

def parseArgs():
  import argparse
  parser = argparse.ArgumentParser(prog=prog['name'], description=description[lang])
  parser.add_argument('--version', action='version', version='%(prog)s {}'.format(prog['version']))
  for i in arguments:
    parser.add_argument(
      arguments[i]["name"],
      default=arguments[i]["default"],
      type=arguments[i]["type"],
      choices=arguments[i]["choices"],
      required=arguments[i]["required"],
      help=arguments[i]["help"][lang],
      metavar=arguments[i]["metavar"],
      dest=arguments[i]["dest"],
    )
  import sys
  if arguments["d"]["name"] in sys.argv:
    args = get_user_inputs()
  else:
    args = vars(parser.parse_args())
  return args

def dataSelection(preds, threshold):
  return preds[preds > threshold]

if __name__ == "__main__":
  import datetime
  start = datetime.datetime.now()
  printInfo(mesages["start"], start.strftime(datetimeFormatOutput))

  # Запрашиваем параметры у пользователя
  user_args = parseArgs()

  from prediction import prepair_model, prepair_data_level1, prepair_data_level2, \
      prepair_dataset, make_predictions, save_rubrics, toRubrics
  from tqdm import tqdm
  import torch
  torch.cuda.empty_cache()

  printInfo(mesages["libs"], datetime.datetime.now().strftime(datetimeFormatOutput))

  model1 = None if models[user_args['modelType']][1] == "" else prepair_model(n_classes=36, lora_model_path=models[user_args['modelType']][1])
  model2 = None
  model3 = None
  if ((user_args['level'] == "RGNTI2") or (user_args['level'] == "RGNTI3")):
      model2 = None if models[user_args['modelType']][2] == "" else prepair_model(n_classes=246, lora_model_path=models[user_args['modelType']][2])
  if (user_args['level'] == "RGNTI3"):
      model3 = None if models[user_args['modelType']][3] == "" else prepair_model(n_classes=0, lora_model_path=models[user_args['modelType']][3])

  if ((model1 is None) or
      ((model2 is None) and ((user_args['level'] == "RGNTI2") or (user_args['level'] == "RGNTI3"))) or
      ((model3 is None) and (user_args['level'] == "RGNTI3"))):
      printInfo(mesages["modelError"], datetime.datetime.now().strftime(datetimeFormatOutput))
      exit()

  printInfo(mesages["startPredict"], datetime.datetime.now().strftime(datetimeFormatOutput))
  df_test = prepair_data_level1(user_args['inFile'], format=user_args['format'])
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  printInfo(mesages["device"], (datetime.datetime.now().strftime(datetimeFormatOutput), device))

  if (user_args['normalisation'] != "not"):
      printInfo(mesages["badFlag"], (datetime.datetime.now().strftime(datetimeFormatOutput), '-n', user_args['normalisation']))
      exit()

  for i in tqdm(range(df_test.shape[0])):
      dataset_loader = prepair_dataset(df_test.iloc[[i]])
      predictions_level1 = make_predictions(model1, dataset_loader, device=device)
      if (user_args['level'] == "RGNTI1"):
          predictions_level1 = toRubrics(os.path.dirname(os.path.abspath(__file__)), predictions_level1, 1, user_args['threshold'])
          save_rubrics(df_test.iloc[[i]], predictions_level1, user_args, prog, i == 0)
      else:
          df_test2 = prepair_data_level2(os.path.dirname(os.path.abspath(__file__)), df_test.iloc[[i]], predictions_level1, user_args['threshold'])
          dataset_loader2 = prepair_dataset(df_test2)
          predictions_level2 = make_predictions(model2, dataset_loader2, device=device)
          if (user_args['level'] == "RGNTI2"):
              predictions_level2 = toRubrics(os.path.dirname(os.path.abspath(__file__)), predictions_level2, 2, user_args['threshold'])
              save_rubrics(df_test2, predictions_level2, user_args, prog, i == 0)
          else:
              printInfo(mesages["notComplete"], datetime.datetime.now().strftime(datetimeFormatOutput))
              predictions_level2 = toRubrics(os.path.dirname(os.path.abspath(__file__)), predictions_level2, 2, user_args['threshold'])
              save_rubrics(df_test2, predictions_level2, user_args, prog, i == 0)

  del model1
  del model2
  del model3

  printInfo(mesages["finish"], datetime.datetime.now().strftime(datetimeFormatOutput))
