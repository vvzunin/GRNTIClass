import json
from messages import printMessage
import os

defaultData = {
  "modelType": "default",
  "models": {
    "default": {
      "textLang": "ru",
      "1": os.path.dirname(os.path.abspath(__file__)) + "\\models\\bert2\\bert_peft_level1",
      "2": os.path.dirname(os.path.abspath(__file__)) + "\\models\\bert2\\bert_peft_level2_with_labels",
      "3": "",
      "description": {
        "en": "Model based on LORA for russian language.",
        "ru": "Модель на основе LORA для русского языка."
      } 
    }
  }
}

class Config():
  def __init__(self):
    self.__dict__ = defaultData
    
  def load(self, file):
    try:
      file = open(file, 'r', encoding="cp1251")
      try:
        data = json.load(file)
        for i in ["modelType", "models"]:
          if i not in data:
            data[i] = defaultData[i]
        data["models"].update(defaultData["models"])
        self.__dict__ = data
      finally:
        file.close()    
    except IOError:
      printMessage("badConfig")
      quit()
      
  def save(self, file):
    with open(file, 'w', encoding='cp1251') as f:
      json.dump(self,
                f,
                default=lambda o: o.__dict__,
                sort_keys=False,
                ensure_ascii=False,
                indent=2)
      
  def print(self):
    for i in self.__dict__:
      print(f'{i}: {self.__dict__[i]}')