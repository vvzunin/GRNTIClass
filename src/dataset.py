import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
import datetime

def convertToDictList(l):
  l1 = []
  l2 = []
  l3 = []
  for rgnti in l:
    s = rgnti.split('.')
    if (len(s) == 3):
      l3.append(s[2])
    else:
      l3.append('')
    if (len(s) >= 2):
      l2.append(s[1])
    else:
      l2.append('')
    l1.append(s[0])    
    
  l1 = '\\'.join(l1)
  l2 = '\\'.join(l2)
  l3 = '\\'.join(l3)
  return {'RGNTI1': l1, 'RGNTI2': l2, 'RGNTI3': l3}

def converDataset(version, lang, dataset_type, print_time = True):
  start_time = datetime.datetime.now()
  in_file = datasets_path + version + "/" + lang + "/raw/" + dataset_type + "_" + lang + ".csv"
  print("Working with " + in_file + ":")
  
  
  df = pd.read_csv(in_file, sep='\t', encoding='cp1251', on_bad_lines='warn')

  # Drop extra columns
  df = df.drop("SUBJ", axis=1)
  df = df.drop("IPV", axis=1)

  # Rename columns
  if (l == "ru"):
    df = df.rename(columns={'id_publ': 'id'})
  elif (l == "eng"):
    df = df.rename(columns={'id_bo': 'id'})
  df = df.rename(columns={'eor': 'correct', 'ref_txt': 'body', 'kw_list': 'keywords'})

  # Split RGNTI column to level colums
  new_df = df["RGNTI"].str.split('\\').apply(lambda x: convertToDictList(x)).apply(pd.Series)
  df = df.drop('RGNTI', axis=1)
  df = pd.concat([df, new_df], axis=1)

  out_dir = datasets_path + version + "/" + lang + "/work/"
  out_file = out_dir + dataset_type + "_" + lang + ".csv"
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  df.to_csv(out_file, sep="\t", index=False, encoding='utf8')
  print("New version saved to " + out_file + "\nTime: {}\n".format(datetime.datetime.now() - start_time))

datasets_path = "../datasets/"
lang = ["ru", "eng"]
dataset_type = ["train", "test"]
versions = ["base"]

start_time = datetime.datetime.now()

for l in lang:
  for v in versions:
    for d in dataset_type:      
      converDataset(v, l, d)

print("Total time: {}".format(datetime.datetime.now() - start_time))
