import json
import os

class TrainSettings():
  def __init__(self):
    self.settings = {
      "curr": ['bert'],
      'all': {
        'BATCH_SIZE': 8,
        'DATASET_NAME': 'base',
        'DATASET_VERSION': 'prep',
        'DATASET_USE_TITLE': True,
        'DATASET_USE_KEYWORDS': False,
        'LANG': 'ru',
        'LEVEL': 'RGNTI1',
        'MIN_TEXTS': {
          'RGNTI1': 5000,
          'RGNTI2': 5000,
          'RGNTI3': 5000
        },
        'MAX_TEXTS': {
          'RGNTI1': 15000,
          'RGNTI2': 15000,
          'RGNTI3': 15000
        },
        'WORKERS': 24
      },
      'bert': {
        'MAX_LEN': 512,
        'PRE_TRAINED_MODEL_NAME': 'DeepPavlov/rubert-base-cased'
      },
      'word2vec': {
        'vector_size': 300,
        'window_size': 4,
        'min_count': 10,
        'skip-gram': True,
        'CBOW': False,
        'hs': 1,
        'negative': 10, # usually between 5 and 20. If 0, not used
        'ns_exponent': 0.75,
        'cbow_mean': 0, # 0 or 1. If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
        'alpha': 0.03, # The initial learning rate.
        'min_alpha': 0.0007, #  Learning rate will linearly drop to min_alpha as training progresses.
        'sample': 6e-5,
        'epochs': 1,
        'compute_loss': True        
      },
      'fasttext': {
        'vector_size': 300,
        'window_size': 4,
        'min_count': 10,
        'skip-gram': True,
        'CBOW': False,
        'hs': 1,
        'negative': 10, # usually between 5 and 20. If 0, not used
        'ns_exponent': 0.75,
        'cbow_mean': 0, # 0 or 1. If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
        'alpha': 0.03, # The initial learning rate.
        'min_alpha': 0.0007, #  Learning rate will linearly drop to min_alpha as training progresses.
        'sample': 6e-5,
        'epochs': 1,
        'compute_loss': True        
      },
      'perceptron': {
        'size': 4
      }
    }

  def save(self, path='', filename="setting.json"):
    if not os.path.exists(path):
      os.makedirs(path)
    with open(path+filename, "w") as outfile:
      s = json.dumps(self.settings, indent=2, sort_keys=False)
      outfile.write(s)

  def load(self, path='', filename="setting.json"):
    with open(path+filename, "r") as infile:
      self.settings = json.loads(infile.read())