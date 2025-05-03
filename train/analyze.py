import os
import json
import csv

def sort_print_top(d, name, top = 5):
  print(name + ":")
  for i, j in (sorted(d.items(), key=lambda x: x[1], reverse=True)[:top]):
    print("\t{}: {:0.5f}".format(i, j))

def analyze(path, res_type="test", top = 5):
  if not os.path.exists(path):
    print("error")
    return
  
  accuracy = {}
  f1_macro = {}
  f1_weighted = {}
  recall_macro = {}
  recall_weighted = {}
  precision_macro = {}
  precision_weighted = {}

  for i in range(0, 108):
    folder = path + "iter_{:03d}/".format(i)
    with open(folder + "report_{}.json".format(res_type), "r") as infile:
      results = json.loads(infile.read())
      accuracy["iter_{:03d}".format(i)] = results["accuracy"]
      f1_macro            ["iter_{:03d}".format(i)] = results["macro avg"]["f1-score"]
      f1_weighted         ["iter_{:03d}".format(i)] = results["weighted avg"]["f1-score"]
      recall_macro        ["iter_{:03d}".format(i)] = results["macro avg"]["recall"]
      recall_weighted     ["iter_{:03d}".format(i)] = results["weighted avg"]["recall"]
      precision_macro     ["iter_{:03d}".format(i)] = results["macro avg"]["precision"]
      precision_weighted  ["iter_{:03d}".format(i)] = results["weighted avg"]["precision"]
  
  sort_print_top(accuracy, "Accuracy", top)
  sort_print_top(f1_macro, "F1. Macro", top)
  sort_print_top(f1_weighted, "F1. Weighted", top)
  sort_print_top(recall_macro, "Recall. Macro", top)
  sort_print_top(recall_weighted, "Recall. Weighted", top)
  sort_print_top(precision_macro, "Precision. Macro", top)
  sort_print_top(precision_weighted, "Precision. Weighted", top)

def settings_csv(path, res_type, model):
  if not os.path.exists(path):
    print("error")
    return
  
  with open('iteration_settings.csv', 'w', newline='') as csvfile:
    fieldnames = ["Name","Models","Type", 
                  "Accuracy",
                  "F1. Macro","F1. Weighted",
                  "Recall. Macro","Recall. Weighted",
                  "Precision. Macro","Precision. Weighted",
                  "vector_size","window_size","min_count",
                  "skip-gram","CBOW","hs","negative",
                  "ns_exponent","cbow_mean","alpha",
                  "min_alpha","sample","epochs","compute_loss",
                  "batch_size","dataset","use_title","use_keywords",
                  "lang","level","min_texts","max_texts"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(216, 324):
      folder = path + "iter_{:03d}/".format(i)
      line = {}
      line["Name"] = model + "_iter_{:03d}".format(i)
      line["Models"] = "LogisticRegression, {}".format(model)
      line["Type"] = res_type

      with open(folder + "report_{}.json".format(res_type), "r") as infile:
        results = json.loads(infile.read())
        line["Accuracy"] = results["accuracy"]
        line["F1. Macro"] = results["macro avg"]["f1-score"]
        line["F1. Weighted"] = results["weighted avg"]["f1-score"]
        line["Recall. Macro"] = results["macro avg"]["recall"]
        line["Recall. Weighted"] = results["weighted avg"]["recall"]
        line["Precision. Macro"] = results["macro avg"]["precision"]
        line["Precision. Weighted"] = results["weighted avg"]["precision"]

      with open(folder + "settings.json", "r") as infile:
        settings = json.loads(infile.read())

        sett = ["vector_size","window_size","min_count",
        "skip-gram","CBOW","hs","negative",
        "ns_exponent","cbow_mean","alpha",
        "min_alpha","sample","epochs","compute_loss"]
        
        for i in sett:
          line[i] = settings[model][i]
        
        line["batch_size"] = settings["all"]["BATCH_SIZE"]
        line["dataset"] = settings["all"]["DATASET_NAME"] + "_" + settings["all"]["DATASET_VERSION"]
        line["use_title"] = settings["all"]["DATASET_USE_TITLE"]
        line["use_keywords"] = settings["all"]["DATASET_USE_KEYWORDS"]
        line["lang"] = settings["all"]["LANG"]
        line["level"] = settings["all"]["LEVEL"]
        line["min_texts"] = settings["all"]["MIN_TEXTS"][settings["all"]["LEVEL"]]
        line["max_texts"] = settings["all"]["MAX_TEXTS"][settings["all"]["LEVEL"]]

        writer.writerow(line)


#analyze("models/fasttext/", "test", 5)
settings_csv("models/word2vec/", "test", "word2vec")