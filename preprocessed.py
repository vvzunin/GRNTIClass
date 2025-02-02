import json
import re
import os
import csv
import statistics
from tqdm import tqdm
import matplotlib.pyplot as plt


# Класс предобработки текста.
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


# Класс предобработки датасетов.
class datasetPreprocessing():
    def __init__(self, tolerance=0.8, dtype=1, lim_min=500):
        self.lim_min = lim_min
        self.tolerance = tolerance
        self.median_rubric = None
        self.dtype = dtype

    # Метод графического вывода количества текстов, соответствующих рубрикам ДО и ПОСЛЕ.
    def review(self, dictOfVal: dict, rebalanced_data: dict) -> None:
        # ДО
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
        plt.xlabel("Количество текстов")
        plt.title("До обработки")

        # ПОСЛЕ
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
        plt.xlabel("Количество текстов")
        median_value = rebalanced_data[self.median_rubric]['amount']
        f_max = ((1 / 2. * self.tolerance + 1) * median_value) / (1 - 1 / 2 * self.tolerance)
        f_min = -f_max + 2 * median_value if -f_max + 2 * median_value >= 0 else 0
        plt.plot([f_min, ((1 / 2. * self.tolerance + 1) * median_value) / (1 - 1 / 2 * self.tolerance)], [0, 0],
                 color="orange", label='Рабочая зона')
        plt.scatter(x, y, color='white', label=f"Rubrics of {self.dtype} type. {l}", linewidth=3, s=90)
        plt.title("После обработки")
        plt.grid(True, linestyle='--', color='gray', alpha=1)
        plt.legend()
        plt.show()

    # Метод загрузки датасета вида raw
    def load_dataset(self, pathToDataSet: str) -> (list, list):
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
        dataDict = [{keys[i]: el for i, el in enumerate(line)} for line in data if len(line) == 8]
        return keys, dataDict

    # Метод расчета количества текстов, соотвующих рубрикам.
    def get_rubrics_amount(self, dataDict: dict) -> dict:
        dictOfValues = {}

        for i in range(len(dataDict)):
            currentData = dataDict[i]

            # Рубрики хранятся под ключом RGNTI, делаем split по разделителю \, получая список рубрик текста
            currentRubricData = currentData["RGNTI"].split("\\")

            # Выделяем рубрику, исходя из типа датасета (dtype). 01.02.03 при dtype=2 -> 01.02
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

        # Возвращаем словарь с количеством текстов у рубрик, где количество текстов >= self.lim_min
        return {x: {k: dictOfValues[x][k] for k in dictOfValues[x]} for x in dictOfValues if
                dictOfValues[x]['amount'] >= self.lim_min}

    # Метод для определения рубрики-медианы.
    def get_median_rubric(self, dictOfValuesNormalized: dict) -> str:

        median = statistics.median([dictOfValuesNormalized[k]['amount'] for k in dictOfValuesNormalized])
        for k in dictOfValuesNormalized:
            if dictOfValuesNormalized[k]['amount'] == median:
                return k

    # Метод, отсеивающий рубрики.
    def rebalance(self, dictOfValues: dict, dictOfValuesNorm: dict) -> dict:
        # Получаем медиану-рубрику
        self.median_rubric = self.get_median_rubric(dictOfValuesNorm)

        if self.median_rubric is None:
            self.median_rubric = list(dictOfValues.keys())[0]
        # Задаем стандартное значение текстов у рубрики по медиане
        standard = dictOfValues[self.median_rubric]['amount']
        print(f"Rubric {self.median_rubric} was peaked as standard. ({standard} el | "
              f"{dictOfValuesNorm[self.median_rubric]['amount']} freq)")
        rebalanced = {}

        # Цикл по рубрикам (k)
        for k in dictOfValues:
            # Данные у рубрики
            currData = dictOfValues[k]
            # Количество текстов у рубрики
            currAmount = currData['amount']

            # Функция погрешности.
            rateAbs = abs(currAmount - standard) / ((standard + currAmount) / 2.)
            rebalanced[k] = {}
            # Если погрешность > значения self.tolerance
            if rateAbs > self.tolerance:

                rebalanced[k]['text'] = []
                # Количество текстов больше границы
                if currAmount >= standard:

                    # Необходимое количество рубрик для удаления
                    amountToDel = int(
                        currAmount - ((1 / 2. * self.tolerance + 1) * standard) / (1 - 1 / 2 * self.tolerance)) + 1
                    # Результативное количество после удаления
                    rebalanced[k]['amount'] = currAmount - amountToDel

                    # Сортировка текстов у рубрики (k) от наибольшего количества рубрик к наименьшему.
                    sortCurrTexts = sorted(
                        currData["text"],
                        key=lambda x: len(set([''.join(l.split(".")[:self.dtype]) for l in x["RGNTI"].split("\\")])),
                        reverse=True)

                    # Тексты к редактированию
                    textsToEdit = sortCurrTexts[:amountToDel]

                    # Оставшиеся тексты
                    textsToAdd = sortCurrTexts[amountToDel:]
                    editedText = []

                    # Цикл удаления рубрик у текстов
                    for i in range(amountToDel):

                        # Получаем все рубрики у текста
                        allRubrics = textsToEdit[i]["RGNTI"].split("\\")

                        # Если количество рубрик равно 1, то мы просто не добавляем текст
                        if len(allRubrics) > 1:
                            if allRubrics.count(k) != len(allRubrics):
                                while k in allRubrics:
                                    # Определяем индекс для удаления. Проходимся по рубрикам в тексте.
                                    indToDel = allRubrics.index(k)

                                    # Удаляем рубрику (k)
                                    allRubrics.pop(indToDel)

                                # Создаем тот же текст без рубрики (k)
                                rgnti = '\\'.join(allRubrics)
                                currText = textsToEdit[i].copy()
                                currText["RGNTI"] = rgnti

                                # Добавляем текст
                                editedText.append(currText)

                    rebalanced[k]['text'] += editedText
                    rebalanced[k]['text'] += textsToAdd

                else:
                    # Количество текстов меньше границы

                    listOfTexts = currData['text']

                    editedText = []
                    # Цикл удаления рубрики у текстов
                    for i in range(len(listOfTexts)):

                        # Получаем все рубрики у текста
                        allRubrics = listOfTexts[i]["RGNTI"].split("\\")

                        # Если количество рубрик равно 1 или одна рубрика в принципе, то мы просто не добавляем текст
                        if len(allRubrics) > 1:
                            if allRubrics.count(k) != len(allRubrics):
                                while k in allRubrics:
                                    # Определяем индекс для удаления. Проходимся по рубрикам в тексте.
                                    indToDel = allRubrics.index(k)

                                    # Удаляем рубрику (k)
                                    allRubrics.pop(indToDel)

                                # Создаем тот же текст без рубрики (k)
                                rgnti = '\\'.join(allRubrics)
                                currText = listOfTexts[i].copy()
                                currText["RGNTI"] = rgnti

                                # Добавляем текст

                                editedText.append(currText)

                    rebalanced[k]['amount'] = 0
                    rebalanced[k]['text'] += editedText
            else:
                # Если погрешность <= значения self.tolerance
                rebalanced[k]['amount'] = dictOfValues[k]["amount"]
                rebalanced[k]['text'] = dictOfValues[k]['text']
        return rebalanced

    # Метод для нормализации от 0 до 1 значений количества текстов у рубрик
    def normalize_dict(self, dictOfValues: dict) -> dict:
        normalizer = sum([dictOfValues[x]['amount'] for x in dictOfValues])
        dictOfValuesNormalized = {k: {"amount": dictOfValues[k]['amount'] / normalizer, "text": dictOfValues[k]['text']}
                                  for k in dictOfValues}
        return dictOfValuesNormalized

    # Преобразование ребалансированной даты.
    def reb_data_toSet(self, data: dict) -> list:
        result_dict = {}

        for k in data:
            try:
                for item in data[k]['text']:
                    title = item['title']
                    rgnti = set(item['RGNTI'].split('\\'))  # Преобразуем в множество

                    # Если текст уже существует, пересекаем значения 'rgnti'
                    if title in result_dict:
                        result_dict[title]['rgnti'] = result_dict[title]['rgnti'].intersection(rgnti)
                    else:
                        # Сохраняем все значения записи, но обрабатываем 'rgnti' отдельно
                        result_dict[title] = {
                            'all_values': item,
                            'rgnti': rgnti
                        }
            except Exception as e:
                print(f"Some Error. Need to fix. {e}")
                continue

        # Преобразуем результат в набор строк
        result_list = list()
        for title in result_dict:
            curr = result_dict[title]['all_values']

            # Заменяем rgnti на конъюнкционное значение
            rgnti = list(result_dict[title]['rgnti'])
            rgnti_al = []

            for rub in rgnti:
                current_rub = '.'.join(rub.split(".")[:self.dtype])
                rgnti_al.append(current_rub)

            rgnti_al = '\\'.join(rgnti_al)
            curr["RGNTI"] = rgnti_al
            curr = list(curr.values())
            result_list.append(curr)

        return result_list

    # Вывод в csv в том же формате raw.
    def get_out(self, data: list, keys: list) -> None:
        print("Start uploading")

        print("Continue")
        with open('output.csv', mode='w', newline='', encoding='windows-1251') as file:
            writer = csv.writer(file, delimiter='\t')  # Устанавливаем табуляцию в качестве разделителя
            writer.writerow(keys)  # Пишем заголовок файла
            for line in tqdm(data, desc="[Uploading]:"):
                writer.writerow(line)  # Пишем строки данных
            print("Done")

    def preprocessed(self, pathToDataset: str) -> None:

        keys, dataDict = self.load_dataset(pathToDataset)

        dictOfValues = self.get_rubrics_amount(dataDict)

        dictOfValuesNorm = self.normalize_dict(dictOfValues)

        rebalancedData = self.rebalance(dictOfValues, dictOfValuesNorm)

        self.review(dictOfValues, rebalancedData)

        listOfReb = self.reb_data_toSet(rebalancedData)

        self.get_out(listOfReb, keys)
