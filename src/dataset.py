import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
import datetime
import re
import matplotlib.pyplot as plt

# from pymorphy2 import MorphAnalyzer
# morph = MorphAnalyzer()

# from nltk.corpus import stopwords

# import swifter
# from swifter import set_defaults
# set_defaults(
#     npartitions=None,
#     dask_threshold=1,
#     scheduler="processes",
#     progress_bar=True,
#     progress_bar_desc=None,
#     allow_dask_on_strings=True,
#     force_parallel=False,
# )

import nltk
# nltk.download('stopwords')
# stopwordsRu = stopwords.words("russian")
# stopwordsEng = stopwords.words("english")

#import spacy
#nlp = spacy.load('en_core_web_sm')
#
#extra_symbols = "[!#$%&'()*+,./:;<=>?@[\]^_`{|}~\"+]"

def lemmatizeRu(doc):
  doc = doc.replace('\\', ' ')
  doc = re.sub("A-Za-z0-9"+extra_symbols, ' ', doc)
  tokens = []
  for token in doc.split():
    if token and token not in stopwordsRu:
      token = token.strip()      
      token = morph.normal_forms(token)[0]            
      tokens.append(token)
  if len(tokens) > 2:
    return ' '.join(tokens)
  return None

def lemmatizeEng(doc):
  doc = doc.replace('\\', ' ')
  doc = re.sub("А-Яа-ЯёЁ0-9"+extra_symbols, ' ', doc)
  morph = spacy.load('en_core_web_sm')
  tokens = []  
  for token in doc.split():
    if token and token not in stopwordsEng:
      token = token.strip()
      doc = nlp(token)      
      token = [tk.lemma_ for tk in doc][0]        
      tokens.append(token)
  if len(tokens) > 2:
    return ' '.join(tokens)
  return None

def prepareText(version, lang, dataset_type, print_time = True):
  start_time = datetime.datetime.now()
  in_file = datasets_path + version + "/" + lang + "/work/" + dataset_type + "_" + lang + ".csv"
  print("Working with " + in_file + ":")
  df = pd.read_csv(in_file, sep='\t', on_bad_lines='warn')

  df['title'] = df['title'].astype('str').swifter.progress_bar(enable=True, desc='title').apply(lemmatizeRu if lang=='ru' else lemmatizeEng)  
  df['body'] = df['body'].astype('str').swifter.progress_bar(enable=True, desc='body').apply(lemmatizeRu if lang=='ru' else lemmatizeEng)  
  df['keywords'] = df['keywords'].astype('str').swifter.progress_bar(enable=True, desc='keywords').apply(lemmatizeRu if lang=='ru' else lemmatizeEng)  

  df = df[df['title'] != '']
  df = df[df['body'] != '']

  out_dir = datasets_path + version + "/" + lang + "/prep/"
  out_file = out_dir + dataset_type + "_" + lang + ".csv"
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  df.to_csv(out_file, sep="\t", index=False, encoding='utf8')
  print("New version saved to " + out_file + "\nTime: {}\n".format(datetime.datetime.now() - start_time))

datasets_path = "datasets/"
lang = ["ru", "eng"]
dataset_type = ["train", "test"]
versions = ["base"]

start_time = datetime.datetime.now()

# TODO: Проблема с английским датасетом
# TODO: Добавить словарь с сокращениями, чтобы оставались.
# TODO: Доработать, чтобы слова типа "3d-принтер" 
# for l in lang:
#   for v in versions:
#     for d in dataset_type:      
#       # converDataset(v, l, d)
#       prepareText(v, l, d)

# print("Total time: {}".format(datetime.datetime.now() - start_time))

in_file = datasets_path + "base" + "/" + "ru" + "/work/" + "train" + "_" + "ru" + ".csv"
df = pd.read_csv(in_file, sep='\t', on_bad_lines='warn')
print(df["RGNTI2"].value_counts().sort_index(ascending=True).to_frame().style.bar().to_excel("test.xlsx"))
# df.groupby(by="RGNTI2").size().plot.hist(by="RGNTI2").get_figure().savefig('example.png')