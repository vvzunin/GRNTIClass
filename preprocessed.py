import json
import re
import os
import csv
import statistics
from tqdm import tqdm
import matplotlib.pyplot as plt


class textPreprocessing():
    def __init__(self):
        self.text = None
        self.preprocessed_text = None
        self.dictOfAbbs = None

    def getText(self, path):
        with open(path, encoding='utf-8') as f:
            a = f.read()
        return a

    def getDictOfAbbs(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            a = json.load(f)
        self.dictOfAbbs = a
        return a

    def replaceAbbs(self):
        dict = self.dictOfAbbs
        text = self.text
        for abbs in dict:
            text = text.replace(abbs, dict[abbs])
        return text

    def remove_latex_formulas(self, text):
        # Удалить формулы внутри $$...$$
        text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
        # Удалить формулы внутри $...$
        text = re.sub(r'\$.*?\$', '', text, flags=re.DOTALL)
        # Удалить формулы внутри \[...\] или \(...\)
        text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
        # Удалить окружения формул (\begin{...}...\end{...})
        text = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', '', text, flags=re.DOTALL)
        return text

    def output(self, pathToDir, outputName):
        text = self.preprocessed_text
        if not text:
            print("No preprocessed data was found")
            return
        if not os.path.isdir(pathToDir):
            print(f"The directory {pathToDir} doesn't exist.\nCreating...")
            os.mkdir(pathToDir)
            print(f"Directory {pathToDir} was created.")
        pathToFile = os.path.join(pathToDir, outputName)
        try:
            with open(pathToFile, 'w', encoding='utf-8') as f:
                f.write(text)
            print("File was successfully writen.")
        except:
            print("Save error. File wasn't written.")

    def preprocessing(self, pathToText, pathToDictOfAbbs=None, replaceFormulas=False):
        self.text = self.getText(pathToText)
        current_text = self.text
        if pathToDictOfAbbs:
            self.dictOfAbbs = self.getDictOfAbbs(pathToDictOfAbbs)
            current_text = self.replaceAbbs()
        if replaceFormulas:
            current_text = self.remove_latex_formulas(current_text)
        self.preprocessed_text = current_text


class datasetPreprocessing():
    def __init__(self, tolerance=0.8, dtype=1, lim_min=500):
        self.lim_min = lim_min
        self.tolerance = tolerance
        self.median_rubric = None
        self.dtype = dtype

    def review(self, dictOfVal, rebalanced_data):
        plt.style.use("dark_background")
        plt.figure(figsize=(20, 15))
        plt.subplot(1, 2, 1)
        x = [float(dictOfVal[k]['amount']) for k in dictOfVal]
        l = len(x)
        y = [0. for _ in dictOfVal]
        median_value = dictOfVal[self.median_rubric]['amount']
        f_max = ((1 / 2. * self.tolerance + 1) * median_value) / (1 - 1 / 2 * self.tolerance)
        f_min = -f_max + 2 * median_value if -f_max + 2 * median_value >= 0 else 0
        plt.plot([f_min, f_max], [0, 0], color="orange", label='Рабочая зона', linewidth=3)
        plt.scatter(x, y, color='white', s=90, label=f"Rubrics of {self.dtype} type. {l}")
        plt.legend()
        plt.grid(True, linestyle='--', color='gray', alpha=1)
        plt.xlabel("Количество рубрик")
        plt.title("До обработки")
        plt.subplot(1, 2, 2)
        reb_data = []
        counter = 0
        for k in rebalanced_data:
            if rebalanced_data[k].get('amount'):
                reb_data.append(rebalanced_data[k]['amount'])
            else:
                counter += 1
        print(f"{counter} el were skipped")
        x = reb_data
        y = [0. for _ in rebalanced_data][counter:]
        l = len(x)
        plt.xlabel("Количество рубрик")
        median_value = rebalanced_data[self.median_rubric]['amount']
        f_max = ((1/2.*self.tolerance+ 1)*median_value)/(1-1/2*self.tolerance)
        f_min = -f_max + 2 * median_value if -f_max + 2 * median_value >=0 else 0
        plt.plot([f_min, ((1/2.*self.tolerance+ 1)*median_value)/(1-1/2*self.tolerance)], [0, 0], color="orange", label='Рабочая зона')
        plt.scatter(x, y, color='white', label=f"Rubrics of {self.dtype} type {l}", linewidth=3, s=90)
        plt.title("После обработки")
        plt.grid(True, linestyle='--', color='gray', alpha=1)
        plt.legend()
        plt.show()

    def load_dataset(self, pathToDataSet):
        print(f"Loading dataset from {pathToDataSet}")
        data = []
        with open(pathToDataSet, mode='r', encoding="windows-1251") as file:
            a = file.read()

        a = a.split("\n")
        for line in a:
            currentData = line.split("\t")
            data.append(currentData)

        if data[-1] == ['']:
            data.pop(-1)

        keys = data.pop(0)
        dataDict = [{keys[i]: el for i, el in enumerate((line))} for line in data]
        return keys, dataDict

    def get_rubrics_amount(self, dataDict):
        dictOfValues = {}

        for i in range(len(dataDict)):
            currentData = dataDict[i]

            currentRubricData = currentData["RGNTI"].split("\\")

            for rubric in currentRubricData:
                if self.dtype != 1:
                    dtypeRubric = '.'.join(rubric.split('.')[:self.dtype])
                else:
                    dtypeRubric = rubric.split('.')[0]
                if dictOfValues.get(dtypeRubric):
                    dictOfValues[dtypeRubric]['amount'] += 1
                    dictOfValues[dtypeRubric]['text'].append(currentData)
                else:
                    dictOfValues[dtypeRubric] = {}
                    dictOfValues[dtypeRubric]['amount'] = 1
                    dictOfValues[dtypeRubric]['text'] = [currentData]
        return {x: {k: dictOfValues[x][k] for k in dictOfValues[x]} for x in dictOfValues if
                dictOfValues[x]['amount'] >= self.lim_min}

    def get_median_rubric(self, dictOfValuesNormalized):

        median = statistics.median([dictOfValuesNormalized[k]['amount'] for k in dictOfValuesNormalized])
        for k in dictOfValuesNormalized:
            if dictOfValuesNormalized[k]['amount'] == median:
                return k

    def rebalance(self, dictOfValues, dictOfValuesNorm):
        self.median_rubric = self.get_median_rubric(dictOfValuesNorm)

        if self.median_rubric is None:
            self.median_rubric = list(dictOfValues.keys())[0]
        standard = dictOfValues[self.median_rubric]['amount']
        print(f"Rubric {self.median_rubric} was peaked as standard. ({standard} el | "
              f"{dictOfValuesNorm[self.median_rubric]['amount']} freq)")
        rebalanced = {}

        for k in dictOfValues:
            currData = dictOfValues[k]
            currAmount = currData['amount']

            rateAbs = abs(currAmount - standard) / ((standard + currAmount) / 2.)
            rebalanced[k] = {}
            if rateAbs > self.tolerance:
                if currAmount >= standard:

                    amountToDel = int(
                        currAmount - ((1/2.*self.tolerance+ 1)*standard)/(1-1/2*self.tolerance)) + 1
                    rebalanced[k]['amount'] = currAmount - amountToDel
                    sortCurrTexts = sorted(
                        currData["text"],
                        key=lambda x: len(set([''.join(l.split(".")[:self.dtype]) for l in x["RGNTI"].split("\\")])),
                        reverse=True)  # Сортировка от наибольшего количества рубрик к наименьшему

                    textsToEdit = sortCurrTexts[:amountToDel]
                    textsToAdd = sortCurrTexts[amountToDel:]
                    editedText = []
                    for i in range(amountToDel):
                        indToDel = None

                        allRubrics = textsToEdit[i]["RGNTI"].split("\\")
                        if len(allRubrics) > 1:
                            for indToDel, rubricsInOne in enumerate(allRubrics):
                                if self.dtype != 1:
                                    curr = ''.join(rubricsInOne.split(".")[:self.dtype])
                                else:
                                    curr = rubricsInOne.split(".")[0]
                                if curr == k:
                                    break
                            allRubrics.pop(indToDel)
                            editedText.append('\\'.join(allRubrics))

                    rebalanced[k]['text'] = textsToAdd
            else:
                rebalanced[k]['amount'] = dictOfValues[k]["amount"]
                rebalanced[k]['text'] = dictOfValues[k]['text']
        return rebalanced

    def normalize_dict(self, dictOfValues):
        normalizer = sum([dictOfValues[x]['amount'] for x in dictOfValues])
        dictOfValuesNormalized = {k: {"amount": dictOfValues[k]['amount'] / normalizer, "text": dictOfValues[k]['text']}
                                  for k in dictOfValues}
        return dictOfValuesNormalized

    def reb_data_toSet(self, data):
        a = set()
        for k in data:
            try:
                for i in range(len(data[k]['text'])):
                    a.add('\t'.join(data[k]['text'][i].values()))
            except Exception as e:
                print(f"Some Error. Need to fix. {e}")
        return a

    def get_out(self, data, keys):
        print("Start uploading")

        print("Continue")
        with open('output.csv', mode='w', newline='', encoding='windows-1251') as file:
            writer = csv.writer(file)

            for line in tqdm(data, desc="[Uploading]:"):
                writer.writerow([line])
            writer.writerow(data)
        print("Done")
    def preprocessed(self, pathToDataset):

        keys, dataDict = self.load_dataset(pathToDataset)

        dictOfValues = self.get_rubrics_amount(dataDict)

        dictOfValuesNorm = self.normalize_dict(dictOfValues)

        rebalancedData = self.rebalance(dictOfValues, dictOfValuesNorm)

        self.review(dictOfValues, dictOfValuesNorm, rebalancedData)
        setData = list(self.reb_data_toSet(rebalancedData))

        self.get_out(setData, keys)
