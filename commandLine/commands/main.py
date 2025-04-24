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
  },
  "p": {
    "name": "-p",
    "default": 1,
    "type": int,
    "choices": None,
    "required": False,
    "help": {
      "ru": "Количество одновременно классифицируемых текстов (по-умолчанию: %(default)s)",
      "en": "Number of simultaneously classified texts (default: %(default)s)"
    },
    "metavar": "packet",
    "dest": "packet"
  },
  "s": {
    "name": "-s",
    "default": False,
    "type": bool,
    "choices": None,
    "required": False,
    "help": {
      "ru": "Включение режима без логов (по-умолчанию: %(default)s)",
      "en": "Enabling No Logs Mode (default: %(default)s)"
    },
    "metavar": "silence",
    "dest": "silence"
  }
}

import click
import sys
import os
import json
from messages import *
from config import *





def loadJSON(jsonPath):
  try:
    file = open(jsonPath, 'r', encoding="cp1251")
    try:
      data = json.load(file)
      return data
    finally:
      file.close()
  except IOError:
    printMessage(messages["badConfig"])
    quit()
  return None
prog = loadJSON("prog.json")
program_version = prog["version"]
# start = datetime.datetime.now()

# messages = messages
# printMessage(messages["start"])

@click.group(invoke_without_command=True, help="test")
@click.option("-s", "--silence", "silence", is_flag=True, help="Включение режима без логов")
@click.option('-v', '--version', is_flag=True, help="Показать версию программы")
@click.pass_context
def main(ctx, silence, version):
  if silence:
    sys.stdout = None
    sys.stderr = None
  if ctx.invoked_subcommand is None:
    click.echo('I was invoked without subcommand')
  else:
    click.echo(f"I am about to invoke {ctx.invoked_subcommand}")
  if version:
     click.echo(f"Версия программы: {program_version}")





@main.group(invoke_without_command=False,
            help="Работа с классификационными моделями моделями")
def models():
  print()
  
@models.command(help="Установка моделей из онлайна")
@click.option('-m', '--model',
              type=str)
def install(model):
  print(model)
  
@models.command(help="Вывод списка моделей")
@click.option('-o', '--online',
              default=False,
              is_flag=True,
              help="Вывод моделей, доступных онлайн")
def list(online):
  if online:
    print("Данный функционал в процессе реализации")
  else:
    path = "config.json"
    if not os.path.exists(path):
        print(f"Файл {path} не найден")
        return

    try:
        with open(path, 'r', encoding='cp1251') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Ошибка при чтении JSON: {e}")
        return

    models_block = data.get("models", {})
    for name, info in models_block.items():
      print("Имя модели:", name)
      print(f'\tЯзык: "{info.get("textLang", "не указано")}"')
      print(f'\tОписание: {info.get("description", "нет описания")}')
      print("\tПути к моделям для классификации:")

      for key, value in info.items():
        if value == '':
          value = "Не указано"
        if key.isdigit():
          print(f"\tУровень {key} - {value}")
@models.command(help="Загрузка конфиг файла")
@click.option('-p', '--path', 'path',
              type=click.Path(exists=True),
              default="config.json",
              help="Путь к конфигурационному файлу")
def config(path):
  config = loadJSON(path)
  models.update(config['models'])
  config['models'] = models


  
@main.group(invoke_without_command=True)


@click.pass_context
def run(ctx):
    """Запуск классификации"""
    if ctx.invoked_subcommand is None:
      click.echo("Команда 'run' без подкоманды. Используйте --help")
  




@run.command(help="Сделать предсказание")
@click.option('-i', '--input', 'input_file', default='text.txt', type=click.Path(exists=True), help='Путь к входному файлу')
@click.option('-o', '--output', 'output_file', default='result.csv', type=click.Path(), help='Путь к результирующий файл')
@click.option('-ei', '--inencode', 'input_encode', default='cp1251', type=str, help='Входна кодировка')
@click.option('-eo', '--outencode', 'output_encode', default='cp1251', type=str, help='Выходная кодировка')
@click.option('-id', '--identifier', 'identifier', default='RGNTI3', 
              type=click.Choice(['RGNTI1', 'RGNTI2', 'RGNTI3']), help='Идентификатор рубрикатора')
@click.option('-p', '--pack', 'packet', default=1, type=int, help='Количество одновременно классифицируемых текстов')
@click.option('-f', '--format', 'input_format', default='plain', type=click.Choice(['plain', 'multidoc']), help='Формат файла')
@click.option('-l', '--lang', 'language', default='ru', type=click.Choice(['ru', 'en']), help='Язык')
@click.option('-t', '--threshold', 'threshold', type=click.FloatRange(0,1),
               default=0.5, help='Минимальная вероятность рубрики (порог)')
@click.option('-n', '--norm', 'normalization', default='not', type=click.Choice(['not', 'some', 'all']), help='Нормализация')
@click.option('-d', '--dialog', 'dialog', help='Диалоговое окно')
def predict(input_file, output_file, input_encode, output_encode, identifier, 
            packet, input_format, language, threshold, normalization, dialog, ):
  if dialog:
    pass
  else:
    from prediction import prepair_model, prepair_data_level1, prepair_data_level2, \
        prepair_dataset, make_predictions, save_rubrics, toRubrics
    from tqdm import tqdm
    import torch
    torch.cuda.empty_cache()
    printMessage(messages["libs"])
    params = {
        "input_file": input_file,
        "output_file": output_file,
        "input_encode": input_encode,
        "output_encode": output_encode,
        "level": identifier,
        "packet": packet,
        "format": input_format,
        "language": language,
        "threshold": threshold,
        "normalisation": normalization
    }
    model1 = None if config['models'][config['modelType']]["1"] == "" \
                  else prepair_model(n_classes=36, 
                                    lora_model_path=config['models'][config['modelType']]["1"])
    model2 = None
    model3 = None
    if ((identifier == "RGNTI2") or (identifier == "RGNTI3")):
        model2 = None if config['models'][config['modelType']]["2"] == ""\
                      else prepair_model(n_classes=246,
                                        lora_model_path=config['models'][config['modelType']]["2"])
    if (identifier == "RGNTI3"):
        model3 = None if config['models'][config['modelType']]["3"] == ""\
                      else prepair_model(n_classes=0,
                                        lora_model_path=config['models'][config['modelType']]["3"])

    if ((model1 is None) or
        ((model2 is None) and ((identifier == "RGNTI2") or (identifier == "RGNTI3"))) or
        ((model3 is None) and (identifier == "RGNTI3"))):
        printMessage(messages["modelError"])
        quit()

    printMessage(messages["startPredict"])
    df_test = prepair_data_level1(input_file, format=input_format, encoding=input_encode)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    printMessage(messages["device"], (device, ))

    if (normalization != "not"):
        printMessage(messages["badFlag"], ('-n', normalization))
        quit()

    for i in tqdm(range(0, df_test.shape[0], packet)):
        dataset_loader = prepair_dataset(df_test.iloc[i:i+packet])
        predictions_level1 = make_predictions(model1, dataset_loader, device=device)
        if (identifier == "RGNTI1"):
            predictions_level1 = toRubrics(os.path.dirname(os.path.abspath(__file__)), predictions_level1, 1, threshold)
            save_rubrics(df_test.iloc[i:i+packet], predictions_level1, prog, i == 0, output_encode)
        else:
            df_test2 = prepair_data_level2(os.path.dirname(os.path.abspath(__file__)), df_test.iloc[i:i+packet], predictions_level1, threshold)
            dataset_loader2 = prepair_dataset(df_test2)
            predictions_level2 = make_predictions(model2, dataset_loader2, device=device)
            if (identifier == "RGNTI2"):
                predictions_level2 = toRubrics(os.path.dirname(os.path.abspath(__file__)), predictions_level2, 2, threshold)
                save_rubrics(df_test2, predictions_level2, params, prog, i == 0, output_encode)
            else:
                printMessage(messages["notComplete"])
                predictions_level2 = toRubrics(os.path.dirname(os.path.abspath(__file__)), predictions_level2, 2, threshold)
                save_rubrics(df_test2, predictions_level2, params, prog, i == 0, output_encode)

    del model1
    del model2
    del model3
  

  
if __name__ == "__main__":
  main()