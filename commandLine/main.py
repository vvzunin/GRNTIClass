prog = {
  'name': 'GRNTIClass',
  'version': '1.3.0'
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
  },
  "badConfig": {
    "ru": "%s Конфиг файл не найден или недоступен",
    "en": "%s Config file not found or unavailable"
  }
}

import os

fileEncodings = ['ascii',
 'big5',
 'big5hkscs',
 'cp037',
 'cp273',
 'cp424',
 'cp437',
 'cp500',
 'cp720',
 'cp737',
 'cp775',
 'cp850',
 'cp852',
 'cp855',
 'cp856',
 'cp857',
 'cp858',
 'cp860',
 'cp861',
 'cp862',
 'cp863',
 'cp864',
 'cp865',
 'cp866',
 'cp869',
 'cp874',
 'cp875',
 'cp932',
 'cp949',
 'cp950',
 'cp1006',
 'cp1026',
 'cp1125',
 'cp1140',
 'cp1250',
 'cp1251',
 'cp1252',
 'cp1253',
 'cp1254',
 'cp1255',
 'cp1256',
 'cp1257',
 'cp1258',
 'euc_jp',
 'euc_jis_2004',
 'euc_jisx0213',
 'euc_kr',
 'gb2312',
 'gbk',
 'gb18030',
 'hz',
 'iso2022_jp',
 'iso2022_jp_1',
 'iso2022_jp_2',
 'iso2022_jp_2004',
 'iso2022_jp_3',
 'iso2022_jp_ext',
 'iso2022_kr',
 'latin_1',
 'iso8859_2',
 'iso8859_3',
 'iso8859_4',
 'iso8859_5',
 'iso8859_6',
 'iso8859_7',
 'iso8859_8',
 'iso8859_9',
 'iso8859_10',
 'iso8859_11',
 'iso8859_13',
 'iso8859_14',
 'iso8859_15',
 'iso8859_16',
 'johab',
 'koi8_r',
 'koi8_t',
 'koi8_u',
 'kz1048',
 'mac_cyrillic',
 'mac_greek',
 'mac_iceland',
 'mac_latin2',
 'mac_roman',
 'mac_turkish',
 'ptcp154',
 'shift_jis',
 'shift_jis_2004',
 'shift_jisx0213',
 'utf_32',
 'utf_32_be',
 'utf_32_le',
 'utf_16',
 'utf_16_be',
 'utf_16_le',
 'utf_7',
 'utf_8',
 'utf_8_sig']

models = {
  "baseModel": {
    "1": os.path.dirname(os.path.abspath(__file__)) + "\\..\\models\\bert2\\bert_peft_level1",
    "2": os.path.dirname(os.path.abspath(__file__)) + "\\..\\models\\bert2\\bert_peft_level2_with_labels",
    "3": "",
    "desc": {
      "ru": "Модель на основе LORA",
      "en": "LORA based model"
    }
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
  "ei": {
    "name": "-ei",
    "default": "cp1251",
    "type": str,
    "choices": fileEncodings,
    "required": False,
    "help": {
      "ru": "Кодировка входного файла (по-умолчанию: %(default)s)",
      "en": "Input file encoding (default: %(default)s)"
    },
    "metavar": "inEncoding",
    "dest": "inEncoding"
  },
  "eo": {
    "name": "-eo",
    "default": "cp1251",
    "type": str,
    "choices": fileEncodings,
    "required": False,
    "help": {
      "ru": "Кодировка выходного файла (по-умолчанию: %(default)s)",
      "en": "Output file encoding (default: %(default)s)"
    },
    "metavar": "outEncoding",
    "dest": "outEncoding"
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
      "ru": "Использованием диалогового взаимодействия с программой (по-умолчанию: %(default)s)",
      "en": "Using dialogue interaction with the program (default: %(default)s)"
    },
    "metavar": "dialogue",
    "dest": "dialogue"
  },
  "c": {
    "name": "-c",
    "default": "config.json",
    "type": str,
    "choices": None,
    "required": False,
    "help": {
      "ru": "Путь к конфигурационному файлу (по-умолчанию: %(default)s)",
      "en": "Path to configuration file (default: %(default)s)"
    },
    "metavar": "config.json",
    "dest": "config"
  }
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

def loadConfig(configPath):
  import json
  try:
    file = open(configPath, 'r', encoding="cp1251")
    try:
      data = json.load(file)
      return data
    finally:
      file.close()
  except IOError:
    printInfo(mesages["badConfig"], datetime.datetime.now().strftime(datetimeFormatOutput))
    quit()
  return None

if __name__ == "__main__":
  import datetime
  start = datetime.datetime.now()
  printInfo(mesages["start"], start.strftime(datetimeFormatOutput))
  
  # Запрашиваем параметры у пользователя
  user_args = parseArgs()
  config = loadConfig(user_args["config"])
  models.update(config['models'])
  config['models'] = models

  from prediction import prepair_model, prepair_data_level1, prepair_data_level2, \
      prepair_dataset, make_predictions, save_rubrics, toRubrics
  from tqdm import tqdm
  import torch
  torch.cuda.empty_cache()

  printInfo(mesages["libs"], datetime.datetime.now().strftime(datetimeFormatOutput))

  model1 = None if models[user_args['modelType']]["1"] == "" else prepair_model(n_classes=36, lora_model_path=models[user_args['modelType']]["1"])
  model2 = None
  model3 = None
  if ((user_args['level'] == "RGNTI2") or (user_args['level'] == "RGNTI3")):
      model2 = None if models[user_args['modelType']]["2"] == "" else prepair_model(n_classes=246, lora_model_path=models[user_args['modelType']]["2"])
  if (user_args['level'] == "RGNTI3"):
      model3 = None if models[user_args['modelType']]["3"] == "" else prepair_model(n_classes=0, lora_model_path=models[user_args['modelType']]["3"])

  if ((model1 is None) or
      ((model2 is None) and ((user_args['level'] == "RGNTI2") or (user_args['level'] == "RGNTI3"))) or
      ((model3 is None) and (user_args['level'] == "RGNTI3"))):
      printInfo(mesages["modelError"], datetime.datetime.now().strftime(datetimeFormatOutput))
      quit()

  printInfo(mesages["startPredict"], datetime.datetime.now().strftime(datetimeFormatOutput))
  df_test = prepair_data_level1(user_args['inFile'], format=user_args['format'], encoding=user_args["inEncoding"])
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  printInfo(mesages["device"], (datetime.datetime.now().strftime(datetimeFormatOutput), device))

  if (user_args['normalisation'] != "not"):
      printInfo(mesages["badFlag"], (datetime.datetime.now().strftime(datetimeFormatOutput), '-n', user_args['normalisation']))
      quit()

  for i in tqdm(range(df_test.shape[0])):
      dataset_loader = prepair_dataset(df_test.iloc[[i]])
      predictions_level1 = make_predictions(model1, dataset_loader, device=device)
      if (user_args['level'] == "RGNTI1"):
          predictions_level1 = toRubrics(os.path.dirname(os.path.abspath(__file__)), predictions_level1, 1, user_args['threshold'])
          save_rubrics(df_test.iloc[[i]], predictions_level1, user_args, prog, i == 0, user_args["outEncoding"])
      else:
          df_test2 = prepair_data_level2(os.path.dirname(os.path.abspath(__file__)), df_test.iloc[[i]], predictions_level1, user_args['threshold'])
          dataset_loader2 = prepair_dataset(df_test2)
          predictions_level2 = make_predictions(model2, dataset_loader2, device=device)
          if (user_args['level'] == "RGNTI2"):
              predictions_level2 = toRubrics(os.path.dirname(os.path.abspath(__file__)), predictions_level2, 2, user_args['threshold'])
              save_rubrics(df_test2, predictions_level2, user_args, prog, i == 0, user_args["outEncoding"])
          else:
              printInfo(mesages["notComplete"], datetime.datetime.now().strftime(datetimeFormatOutput))
              predictions_level2 = toRubrics(os.path.dirname(os.path.abspath(__file__)), predictions_level2, 2, user_args['threshold'])
              save_rubrics(df_test2, predictions_level2, user_args, prog, i == 0, user_args["outEncoding"])

  del model1
  del model2
  del model3

  printInfo(mesages["finish"], datetime.datetime.now().strftime(datetimeFormatOutput))
