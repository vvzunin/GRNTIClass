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
from config import *

@click.group(invoke_without_command=True, help="test")
@click.pass_context
def main(ctx):
  if ctx.invoked_subcommand is None:
    click.echo('I was invoked without subcommand')
  else:
    click.echo(f"I am about to invoke {ctx.invoked_subcommand}")




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
    pass
  
  
  
@main.group(invoke_without_command=False,
            chain=True,
            help="Запуск классификации")
def run():
  print()
  
@run.command(help="Загрузка конфиг файла")
@click.option('-p', '--path', 'path',
              type=str,
              default="config.json",
              help="Путь к конфигурационному файлу")
def config():
  print()
  
  
@main.group(invoke_without_command=False,
            chain=True,
            help="Запуск классификации")
def run():
  print()
  
@run.command(help="Загрузка конфиг файла")
@click.option('-p', '--path', 'path',
              type=str,
              default="config.json",
              help="Путь к конфигурационному файлу")
def config():
  print()
  
if __name__ == "__main__":
  main()