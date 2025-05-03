fileEncodings = [
  "ascii",
  "big5",
  "big5hkscs",
  "cp037",
  "cp273",
  "cp424",
  "cp437",
  "cp500",
  "cp720",
  "cp737",
  "cp775",
  "cp850",
  "cp852",
  "cp855",
  "cp856",
  "cp857",
  "cp858",
  "cp860",
  "cp861",
  "cp862",
  "cp863",
  "cp864",
  "cp865",
  "cp866",
  "cp869",
  "cp874",
  "cp875",
  "cp932",
  "cp949",
  "cp950",
  "cp1006",
  "cp1026",
  "cp1125",
  "cp1140",
  "cp1250",
  "cp1251",
  "cp1252",
  "cp1253",
  "cp1254",
  "cp1255",
  "cp1256",
  "cp1257",
  "cp1258",
  "euc_jp",
  "euc_jis_2004",
  "euc_jisx0213",
  "euc_kr",
  "gb2312",
  "gbk",
  "gb18030",
  "hz",
  "iso2022_jp",
  "iso2022_jp_1",
  "iso2022_jp_2",
  "iso2022_jp_2004",
  "iso2022_jp_3",
  "iso2022_jp_ext",
  "iso2022_kr",
  "latin_1",
  "iso8859_2",
  "iso8859_3",
  "iso8859_4",
  "iso8859_5",
  "iso8859_6",
  "iso8859_7",
  "iso8859_8",
  "iso8859_9",
  "iso8859_10",
  "iso8859_11",
  "iso8859_13",
  "iso8859_14",
  "iso8859_15",
  "iso8859_16",
  "johab",
  "koi8_r",
  "koi8_t",
  "koi8_u",
  "kz1048",
  "mac_cyrillic",
  "mac_greek",
  "mac_iceland",
  "mac_latin2",
  "mac_roman",
  "mac_turkish",
  "ptcp154",
  "shift_jis",
  "shift_jis_2004",
  "shift_jisx0213",
  "utf_32",
  "utf_32_be",
  "utf_32_le",
  "utf_16",
  "utf_16_be",
  "utf_16_le",
  "utf_7",
  "utf_8",
  "utf_8_sig",
]

import click
import sys
import os
import json
from messages import *
from config import *

def loadJSON(jsonPath):
  try:
    file = open(jsonPath, "r", encoding="cp1251")
    try:
      data = json.load(file)
      return data
    finally:
      file.close()
  except IOError:
    printMessage("badConfig")
    quit()
  return None

configPath = "config.json"
progPath = "prog.json"
prog = loadJSON(progPath)
lang = prog["language"]

@click.group(
  invoke_without_command=True,
  help=prog["description"][prog["language"]].format(prog["name"],
                            prog["version"]),
)
@click.option(
  "-v",
  "--version",
  is_flag=True,
  help="Показать версию программы")
@click.option(
  "-l",
  "--lang",
  "lang",
  type=click.Choice(["ru", "en"]),
  help="Смена языка интерфейса программы",
)
@click.pass_context
def main(ctx, version, lang):
  if version:
    click.echo(prog["version"])
  elif lang:
    prog["language"] = lang
    with open(progPath, "w", encoding="cp1251") as f:
      json.dump(prog, f, indent=2, sort_keys=True, ensure_ascii=False)
  else:
    if ctx.invoked_subcommand is None:
      click.echo("Программа запущена без комманд и флагов!")


@main.group(
  invoke_without_command=False, help="Работа с классификационными моделями моделями"
)
def models():
  pass


@models.command(help="Загрузка конфиг файла")
@click.option(
  "-p",
  "--path",
  "path",
  type=click.Path(exists=True),
  default="config.json",
  help="Путь к конфигурационному файлу",
)
def config(path):
  prog["configPath"] = path
  with open(progPath, "w", encoding="cp1251") as f:
    json.dump(prog, f, indent=2, sort_keys=True, ensure_ascii=False)
  config = loadJSON(prog["configPath"])
  

@models.command(help="Установка моделей из онлайна")
@click.option("-m", "--model", type=str)
def install(model):
  print(model)


@models.command(help="Вывод списка моделей")
@click.option(
  "-o",
  "--online",
  default=False,
  is_flag=True,
  help="Вывод моделей, доступных онлайн",
)
def list(online):
  if online:
    printMessage("notComplete")
  else:
    data = loadJSON(prog["configPath"])

    models_block = data.get("models", {})
    for name, info in models_block.items():
      print("Имя модели:", name)
      print(f'\tЯзык: "{info.get("textLang", "не указано")}"')
      print(f'\tОписание: {info.get("description", "нет описания")}')
      print("\tПути к моделям для классификации:")

      for key, value in info.items():
        if value == "":
          value = "Не указано"
        if key.isdigit():
          print(f"\tУровень {key} - {value}")


@main.command(help="Сделать предсказание")
@click.option(
  "-i",
  "--input",
  "input_file",
  default="text.txt",
  type=click.Path(exists=True),
  help="Путь к входному файлу",
)
@click.option(
  "-o",
  "--output",
  "output_file",
  default="result.csv",
  type=click.Path(),
  help="Путь к результирующий файл",
)
@click.option(
  "-ei",
  "--inencode",
  "input_encode",
  default="cp1251",
  type=str,
  help="Входна кодировка",
)
@click.option(
  "-eo",
  "--outencode",
  "output_encode",
  default="cp1251",
  type=str,
  help="Выходная кодировка",
)
@click.option(
  "-id",
  "--identifier",
  "identifier",
  default="RGNTI3",
  type=click.Choice(["RGNTI1", "RGNTI2", "RGNTI3"]),
  help="Идентификатор рубрикатора",
)
@click.option(
  "-p",
  "--pack",
  "packet",
  default=1,
  type=int,
  help="Размер пакета текстов для классификации",
)
@click.option(
  "-f",
  "--format",
  "input_format",
  default="plain",
  type=click.Choice(["plain", "multidoc"]),
  help="Формат файла",
)
@click.option(
  "-l",
  "--lang",
  "language",
  default="ru",
  type=click.Choice(["ru", "en"]),
  help="Язык",
)
@click.option(
  "-t",
  "--threshold",
  "threshold",
  type=click.FloatRange(0, 1),
  default=0.5,
  help="Минимальная вероятность рубрики (порог)",
)
@click.option(
  "-n",
  "--norm",
  "normalization",
  default="not",
  type=click.Choice(["not", "some", "all"]),
  help="Нормализация",
)
@click.option(
  "-dv",
  "--device",
  "device",
  type=click.Choice(["cpu", "cuda:0"]),
  default="cpu",
  help="Выбор устройства выполнения",
)
@click.option("-d", "--dialog", "dialog", help="Диалоговое окно")
@click.option(
  "-s", "--silence", "silence", is_flag=True, help="Включение режима без логов"
)
@click.option(
  "-w",
  "--workers",
  "workers",
  default=1,
  type=click.INT,
  help="Максимальное количество процессов для загрузки данных",
)
def predict(
  input_file,
  output_file,
  input_encode,
  output_encode,
  identifier,
  packet,
  input_format,
  language,
  threshold,
  normalization,
  device,
  dialog,
  silence,
  workers,
):
  if silence:
    sys.stdout = None
    sys.stderr = None
  start = datetime.datetime.now()
  printMessage("start")
  if dialog:
    print("Development in progress!")
  else:
    from prediction import (
      prepair_model,
      prepair_data_level1,
      prepair_data_level2,
      prepair_dataset,
      make_predictions,
      save_rubrics,
      toRubrics,
    )
    from tqdm import tqdm
    import torch

    torch.cuda.empty_cache()
    printMessage("libs")
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
      "normalisation": normalization,
      "device": device,
      "workers": workers,
    }
    config = loadJSON(configPath)
    model1 = (
      None
      if config["models"][config["modelType"]]["1"] == ""
      else prepair_model(
        n_classes=36, lora_model_path=config["models"][config["modelType"]]["1"]
      )
    )
    model2 = None
    model3 = None
    if (identifier == "RGNTI2") or (identifier == "RGNTI3"):
      model2 = (
        None
        if config["models"][config["modelType"]]["2"] == ""
        else prepair_model(
          n_classes=246,
          lora_model_path=config["models"][config["modelType"]]["2"],
        )
      )
    if identifier == "RGNTI3":
      model3 = (
        None
        if config["models"][config["modelType"]]["3"] == ""
        else prepair_model(
          n_classes=0,
          lora_model_path=config["models"][config["modelType"]]["3"],
        )
      )

    if (
      (model1 is None)
      or (
        (model2 is None)
        and ((identifier == "RGNTI2") or (identifier == "RGNTI3"))
      )
      or ((model3 is None) and (identifier == "RGNTI3"))
    ):
      printMessage("modelError")
      quit()

    printMessage("startPredict")
    df_test = prepair_data_level1(
      input_file, format=input_format, encoding=input_encode
    )
    device = torch.device(params["device"] if torch.cuda.is_available() else "cpu")
    printMessage("device", "ru", (device,))

    if normalization != "not":
      printMessage("badFlag", "ru", ("-n", normalization))
      quit()

    for i in tqdm(range(0, df_test.shape[0], packet)):
      dataset_loader = prepair_dataset(
        df_test.iloc[i : i + packet], workers=params["workers"]
      )
      predictions_level1 = make_predictions(model1, dataset_loader, device=device)
      if identifier == "RGNTI1":
        predictions_level1 = toRubrics(
          config["models"][config["modelType"]]["1"],
          predictions_level1,
          threshold,
        )
        save_rubrics(
          df_test.iloc[i : i + packet],
          predictions_level1,
          params,
          prog,
          i == 0,
          output_encode,
        )
      else:
        df_test2 = prepair_data_level2(
          os.path.dirname(os.path.abspath(__file__)),
          config["models"][config["modelType"]]["1"],
          df_test.iloc[i : i + packet],
          predictions_level1,
          threshold,
        )
        dataset_loader2 = prepair_dataset(df_test2, workers=params["workers"])
        predictions_level2 = make_predictions(
          model2, dataset_loader2, device=device
        )
        if identifier == "RGNTI2":
          predictions_level2 = toRubrics(
            config["models"][config["modelType"]]["2"],
            predictions_level2,
            threshold,
          )
          save_rubrics(
            df_test2,
            predictions_level2,
            params,
            prog,
            i == 0,
            output_encode,
          )
        else:
          printMessage("notComplete")
          predictions_level2 = toRubrics(
            config["models"][config["modelType"]]["2"],
            predictions_level2,
            threshold,
          )
          save_rubrics(
            df_test2,
            predictions_level2,
            params,
            prog,
            i == 0,
            output_encode,
          )

    del model1
    del model2
    del model3
    printMessage("finish")


@main.command(help="Запустить сервер")
@click.option("-h", "--host", "host", default="localhost", help="IP адрес")
@click.option("-p", "--port", "port", default=8000, type=click.INT, help="Порт")
@click.option(
  "-dv",
  "--device",
  "device",
  type=click.Choice(["cpu", "cuda:0"]),
  default="cpu",
  help="Выбор устройства выполнения",
)
@click.option(
  "-s", "--silence", "silence", is_flag=True, help="Включение режима без логов"
)
def server(host, port, device, silence):
  pass


if __name__ == "__main__":
  main()
