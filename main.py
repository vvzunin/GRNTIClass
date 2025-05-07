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
from help_message import get_help

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

progPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prog.json")
prog = loadJSON(progPath)

def show_help(ctx, param, value):
    if value:
        ctx.exit(ctx.get_help())

def help_flag(func):
    return click.option('--help', 
                        is_flag=True, 
                        expose_value=False, 
                        is_eager=True, 
                        callback=show_help, 
                        help=get_help("help", prog["language"]))(func)

@click.group(
  invoke_without_command=True,
  help=prog["description"][prog["language"]].format(prog["name"],
                            prog["version"]),
)
@click.option(
  "-v",
  "--version",
  is_flag=True,
  help=get_help("version", prog["language"]))
@click.option(
  "-l",
  "--lang",
  "lang",
  type=click.Choice(["ru", "en"]),
  help=get_help("lang", prog["language"]),
)
@help_flag
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
      click.echo(get_help("main_description", prog["language"]))


@main.group(
  invoke_without_command=False, help=get_help("models", prog["language"])
)
@help_flag
def models():
  pass


@models.command(help=get_help("config", prog["language"]))
@click.option(
  "-p",
  "--path",
  "path",
  type=click.Path(exists=True),
  default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
  help=get_help("config_path", prog["language"]),
)
@help_flag
def config(path):
  prog["configPath"] = path
  with open(progPath, "w", encoding="cp1251") as f:
    json.dump(prog, f, indent=2, sort_keys=True, ensure_ascii=False)
  config = loadJSON(prog["configPath"])
  

@models.command(help=get_help("install", prog["language"]))
@click.option("-m", "--model", type=str, help=get_help("model", prog["language"]))
@help_flag
def install(model):
  print(model)


@models.command(help=get_help("list", prog["language"]))
@click.option(
  "-o",
  "--online",
  default=False,
  is_flag=True,
  help=get_help("list_online", prog["language"]),
)
@help_flag
def list_models(online):
  if online:
    printMessage("notComplete")
    return

  data = loadJSON(prog["configPath"])
  models_block = data.get("models", {})

  for name, info in models_block.items():
    if prog["language"] == "en":
      print("Model name:", name)
      print(f'\tLanguage: {info.get("textLang", "not specified")}')
      print(f'\tDescription: {info.get("description", "no description")[prog["language"]]}'),
      print("\tModels' paths for classification:")
    else:
      print("Имя модели:", name)
      print(f'\tЯзык: {info.get("textLang", "не указано")}')
      print(f'\tОписание: {info.get("description", "нет описания")[prog["language"]]}')
      print("\tПути к моделям для классификации:")

    for key, value in info.items():
      if key.isdigit():
        if value == "":
          value = "Not specified" if prog["language"] == "en" else "Не указано"
        if prog["language"] == "en":
          print(f"\t    Level {key} - {value}")
        else:
          print(f"\t    Уровень {key} - {value}")



@main.command(help=get_help("predict", prog["language"]))
@click.option(
  "-i",
  "--input",
  "input_file",
  default="text.txt",
  type=click.Path(exists=True),
  help=get_help("input_file", prog["language"]),
)
@click.option(
  "-o",
  "--output",
  "output_file",
  default="result.csv",
  type=click.Path(),
  help=get_help("output_file", prog["language"]),
)
@click.option(
  "-ei",
  "--inencode",
  "input_encode",
  default="cp1251",
  type=str,
  help=get_help("input_encode", prog["language"]),
)
@click.option(
  "-eo",
  "--outencode",
  "output_encode",
  default="cp1251",
  type=str,
  help=get_help("output_encode", prog["language"]),
)
@click.option(
  "-id",
  "--identifier",
  "identifier",
  default="RGNTI3",
  type=click.Choice(["RGNTI1", "RGNTI2", "RGNTI3"]),
  help=get_help("identifier", prog["language"]),
)
@click.option(
  "-p",
  "--pack",
  "packet",
  default=1,
  type=int,
  help=get_help("packet", prog["language"]),
)
@click.option(
  "-f",
  "--format",
  "input_format",
  default="plain",
  type=click.Choice(["plain", "multidoc"]),
  help=get_help("input_format", prog["language"]),
)
@click.option(
  "-l",
  "--lang",
  "language",
  default="ru",
  type=click.Choice(["ru", "en"]),
  help=get_help("language", prog["language"]),
)
@click.option(
  "-t",
  "--threshold",
  "threshold",
  type=click.FloatRange(0, 1),
  default=0.5,
  help=get_help("threshold", prog["language"]),
)
@click.option(
  "-n",
  "--norm",
  "normalization",
  default="not",
  type=click.Choice(["not", "some", "all"]),
  help=get_help("normalization", prog["language"]),
)
@click.option(
  "-dv",
  "--device",
  "device",
  type=click.Choice(["cpu", "cuda:0"]),
  default="cpu",
  help=get_help("device", prog["language"]),
)
@click.option("-d", "--dialog", "dialog", help=get_help("dialog", prog["language"]))
@click.option(
  "-s", "--silence", "silence", is_flag=True, help=get_help("silence", prog["language"])
)
@click.option(
  "-w",
  "--workers",
  "workers",
  default=1,
  type=click.INT,
  help=get_help("workers", prog["language"]),
)
@help_flag
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
    config = loadJSON(prog["configPath"])
    print(config["models"][config["modelType"]]["1"])
    model1 = (
      None
      if config["models"][config["modelType"]]["1"] == ""
      else prepair_model(
        n_classes=36, lora_model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), config["models"][config["modelType"]]["1"])
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
          lora_model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), config["models"][config["modelType"]]["2"])
        )
      )
    if identifier == "RGNTI3":
      model3 = (
        None
        if config["models"][config["modelType"]]["3"] == ""
        else prepair_model(
          n_classes=0,
          lora_model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), config["models"][config["modelType"]]["3"]),
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
          os.path.join(os.path.dirname(os.path.abspath(__file__)), config["models"][config["modelType"]]["1"]),
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
          os.path.join(os.path.dirname(os.path.abspath(__file__)), config["models"][config["modelType"]]["1"]),
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
            os.path.join(os.path.dirname(os.path.abspath(__file__)), config["models"][config["modelType"]]["2"]),
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
            os.path.join(os.path.dirname(os.path.abspath(__file__)), config["models"][config["modelType"]]["2"]),
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

default_config = {
    "api": {
        "docker_host": "0.0.0.0",
        "local_host": "localhost",
        "port": 8000
    },
    "device": "cpu"
}

@main.command(help=get_help("server", prog["language"]))
@click.option("-h", "--host", "host", default="localhost", help=get_help("host", prog["language"]))
@click.option("-p", "--port", "port", default=8000, type=click.INT, help=get_help("port", prog["language"]))
@click.option(
  "-dv",
  "--device",
  "device",
  type=click.Choice(["cpu", "cuda:0"]),
  default="cpu",
  help=get_help("device", prog["language"]),
)
@click.option(
  "-s", "--silence", "silence", is_flag=True, help=get_help("silence", prog["language"])
)
@help_flag
def server(host, port, device, silence):
  if silence:
    sys.stdout = None
    sys.stderr = None 

  ends = ["backend", "frontend"]
  for end in ends:
    config_path = os.path.join(os.path.dirname(__file__), "src", end, "config.json")
    if not os.path.exists(config_path):
      with open(config_path, "w") as f:
        json.dump(default_config, f, indent=2)
    
    config = loadJSON(config_path)
    config["api"]["local_host"] = host
    config["api"]["port"] = port
    config["device"] = device

    with open(config_path, "w", encoding="utf-8") as file:
      json.dump(config, file, indent=2, ensure_ascii=False)

  from src.backend.main import backend_startup
  
  backend_startup()

from multiprocessing import freeze_support

if __name__ == "__main__":
  freeze_support()
  main()
