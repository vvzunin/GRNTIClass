import json

class TrainSettings():
  def __init__(self):
    self.settings = {
      "curr": ['bert'],
      'all': {
        'BATCH_SIZE': 8,
        'DATASET_NAME': 'base'
      },
      'bert': {
        'MAX_LEN': 512,        
        'PRE_TRAINED_MODEL_NAME': 'DeepPavlov/rubert-base-cased'     
      },
      'word2vec': {
        'embedding_size': 64,
        'window_size': 4,
        'num_sampled': 32
      }
    }

  def save(self, filename="setting.json"):
    with open(filename, "w") as outfile:
      s = json.dumps(self.settings, indent=2, sort_keys=False)
      outfile.write(s)

  def load(self, filename="setting.json"):
    with open(filename, "r") as infile:
      self.settings = json.loads(infile.read())