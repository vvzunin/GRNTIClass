import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os

datasets_path = "../datasets/"
lang = ["ru", "eng"]
dataset_type = ["train", "test"]
versions = ["base"]

for l in lang:
  for v in versions:
    for d in dataset_type:
      in_file = datasets_path + v + "/" + l + "/raw/" + d + "_" + l + ".csv"
      print("Working with " + in_file + ":")
      
      df = pd.read_csv(in_file, sep='\t', encoding='cp1251', on_bad_lines='warn')

      df = df.drop("SUBJ", axis=1)
      df = df.drop("IPV", axis=1)
      if (l == "ru"):
        df = df.rename(columns={'id_publ': 'id'})
      elif (l == "eng"):
        df = df.rename(columns={'id_bo': 'id'})
      df = df.rename(columns={'eor': 'correct', 'ref_txt': 'body', 'kw_list': 'keywords'})

      out_dir = datasets_path + v + "/" + l + "/work/"
      out_file = out_dir + d + "_" + l + ".csv"
      if not os.path.exists(out_dir):
        os.makedirs(out_dir)
      df.to_csv(out_file, sep="\t", index=False)
      print("Complete! New version saved to " + out_file + "\n")