import json

class TrainSettings():
  def __init__(self, MAX_LEN=512, BATCH_SIZE=8, PRE_TRAINED_MODEL_NAME='DeepPavlov/rubert-base-cased'):
    self.settings = {
      'MAX_LEN': MAX_LEN,
      'BATCH_SIZE': BATCH_SIZE,
      'PRE_TRAINED_MODEL_NAME': PRE_TRAINED_MODEL_NAME
    }

  def save(self, filename="setting.json"):
    with open(filename, "w") as outfile:
      s = json.dumps(self.settings, indent=2, sort_keys=False)
      outfile.write(s)

  def load(self, filename="setting.json"):
    with open(filename, "r") as infile:
      self.settings = json.loads(infile.read())