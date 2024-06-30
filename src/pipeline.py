from TrainSettings import TrainSettings
from PrepareData import PrepareData
from VectorizeWord2Vec import VectorizeWord2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os
import json
from tqdm import tqdm

sett = TrainSettings()
sett.settings['curr'] = ['Word2Vec', 'LogisticRegression']

def calc(settings, type, path, dataset_path):
  prepdata = PrepareData(sett.settings['all'], dataset_path)
  (compYtoRGNTI, compRGNTItoY) = prepdata.prepareAll()

  vect = VectorizeWord2Vec(sett.settings[type], type, path, sett.settings['all']['WORKERS'])
  vect.build_vocab(prepdata.df_train['text'])
  vect.train(prepdata.df_train['text'], save = False)

  X_train = vect.vectorizeAll(prepdata.df_train['text'])
  X_test  = vect.vectorizeAll(prepdata.df_test['text'])
  
  (y_train, y_test) = prepdata.toCat()
  #print(y_test.apply(lambda x: compYtoRGNTI[x] if x in compYtoRGNTI else 'Unknown'))

  model = LogisticRegression(n_jobs = sett.settings['all']['WORKERS'], max_iter=100)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_train)
  report = classification_report(y_train, y_pred, output_dict=True)

  if not os.path.exists(path):
      os.makedirs(path)
  with open(path + "report_train.json", "w") as outfile:
    s = json.dumps(report, indent=2, sort_keys=False)
    outfile.write(s)

  y_pred = model.predict(X_test)
  report = classification_report(y_test, y_pred, output_dict=True)
  with open(path + "report_test.json", "w") as outfile:
    s = json.dumps(report, indent=2, sort_keys=False)
    outfile.write(s)
  
  sett.save(path,"settings.json")


n = 216
with tqdm(total=216) as pbar:
  for type in ['word2vec', 'fasttext']:
    for DATASET_USE_TITLE in [True]:
      for DATASET_USE_KEYWORDS  in [True]:
          for LEVEL in ['RGNTI2']: #, 'RGNTI3']:
            for MIN_TEXTS in [500]:
                for MAX_TEXTS in [5000]:
                  for vector_size in [200, 250, 300]:
                      for window_size in [2, 4, 6]:
                        for min_count in [10]:
                            for skip_gram in [False, True]:
                              for hs in [0, 1]:
                                ng = [0, 10] if hs == 1 else [10]
                                for negative in ng:
                                    for alpha in [0.03]:
                                      for epochs in [5, 10]:
                                        sett.settings['all']['DATASET_USE_TITLE'] = DATASET_USE_TITLE
                                        sett.settings['all']['DATASET_USE_KEYWORDS'] = DATASET_USE_KEYWORDS
                                        sett.settings['all']['LEVEL'] = LEVEL
                                        sett.settings['all']['MIN_TEXTS'][LEVEL] = MIN_TEXTS
                                        sett.settings['all']['MAX_TEXTS'][LEVEL] = MAX_TEXTS
                                        sett.settings[type]['vector_size'] = vector_size
                                        sett.settings[type]['window_size'] = window_size
                                        sett.settings[type]['skip-gram'] = skip_gram
                                        sett.settings[type]['hs'] = hs
                                        sett.settings[type]['negative'] = negative
                                        sett.settings[type]['alpha'] = alpha
                                        sett.settings[type]['epochs'] = epochs
                                        calc(sett, type, "../models/{}/iter_{:03d}/".format(type, n) , "../datasets/")
                                          
                                        n += 1
                                        pbar.update(1)
print(n-216)