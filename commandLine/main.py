import os
from messages import printMessage

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

lang = "ru"

def dataSelection(preds, threshold):
  return preds[preds > threshold]

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
    
if __name__ == "__main__":
  import datetime, sys
    
  # Запрашиваем параметры у пользователя
  from commands.main import main as cli
  
  import json
  prog = loadJSON("prog.json")
  
  cli()
    
  quit()
  
  user_args = []
  
  if (user_args["silence"]):
    sys.stdout = None
    sys.stderr = None
  
  start = datetime.datetime.now()
  
  messages = loadJSON("messages.json")
  printMessage(messages["start"])
  
  from config import config
  config = loadJSON(user_args["config"])
  models.update(config['models'])
  config['models'] = models

  from prediction import prepair_model, prepair_data_level1, prepair_data_level2, \
      prepair_dataset, make_predictions, save_rubrics, toRubrics
  from tqdm import tqdm
  import torch
  torch.cuda.empty_cache()

  printMessage(messages["libs"])

  model1 = None if config['models'][config['modelType']]["1"] == "" \
                else prepair_model(n_classes=36, 
                                   lora_model_path=config['models'][config['modelType']]["1"])
  model2 = None
  model3 = None
  if ((user_args['level'] == "RGNTI2") or (user_args['level'] == "RGNTI3")):
      model2 = None if config['models'][config['modelType']]["2"] == ""\
                    else prepair_model(n_classes=246,
                                       lora_model_path=config['models'][config['modelType']]["2"])
  if (user_args['level'] == "RGNTI3"):
      model3 = None if config['models'][config['modelType']]["3"] == ""\
                    else prepair_model(n_classes=0,
                                       lora_model_path=config['models'][config['modelType']]["3"])

  if ((model1 is None) or
      ((model2 is None) and ((user_args['level'] == "RGNTI2") or (user_args['level'] == "RGNTI3"))) or
      ((model3 is None) and (user_args['level'] == "RGNTI3"))):
      printMessage(messages["modelError"])
      quit()

  printMessage(messages["startPredict"])
  df_test = prepair_data_level1(user_args['inFile'], format=user_args['format'], encoding=user_args["inEncoding"])
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  printMessage(messages["device"], (device, ))

  if (user_args['normalisation'] != "not"):
      printMessage(messages["badFlag"], ('-n', user_args['normalisation']))
      quit()

  for i in tqdm(range(0, df_test.shape[0], user_args['packet'])):
      dataset_loader = prepair_dataset(df_test.iloc[i:i+user_args['packet']])
      predictions_level1 = make_predictions(model1, dataset_loader, device=device)
      if (user_args['level'] == "RGNTI1"):
          predictions_level1 = toRubrics(os.path.dirname(os.path.abspath(__file__)), predictions_level1, 1, user_args['threshold'])
          save_rubrics(df_test.iloc[i:i+user_args['packet']], predictions_level1, user_args, prog, i == 0, user_args["outEncoding"])
      else:
          df_test2 = prepair_data_level2(os.path.dirname(os.path.abspath(__file__)), df_test.iloc[i:i+user_args['packet']], predictions_level1, user_args['threshold'])
          dataset_loader2 = prepair_dataset(df_test2)
          predictions_level2 = make_predictions(model2, dataset_loader2, device=device)
          if (user_args['level'] == "RGNTI2"):
              predictions_level2 = toRubrics(os.path.dirname(os.path.abspath(__file__)), predictions_level2, 2, user_args['threshold'])
              save_rubrics(df_test2, predictions_level2, user_args, prog, i == 0, user_args["outEncoding"])
          else:
              printMessage(messages["notComplete"])
              predictions_level2 = toRubrics(os.path.dirname(os.path.abspath(__file__)), predictions_level2, 2, user_args['threshold'])
              save_rubrics(df_test2, predictions_level2, user_args, prog, i == 0, user_args["outEncoding"])

  del model1
  del model2
  del model3

  printMessage(messages["finish"])
