import os
import csv
import statistics
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
import torch
from peft import TaskType, LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification,\
    AutoTokenizer, DataCollatorWithPadding
from collections import Counter
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MultilabelF1Score, MultilabelAccuracy,\
      MultilabelAUROC
from datasets import Dataset



import seaborn as sns

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

    def get_grnti1_3_BERT_dataframes(self, file_path, number_of_delteted_values,
                                     minimal_number_of_elements_RGNTI3,
                                     minimal_number_of_words,
                                     dir_name=None,
                                     change_codes=False,
                                     grnti_folder=""):

        # Чтение исходных данных
        df = pd.read_csv(file_path + "/train_ru.csv", sep='\t', encoding='cp1251')
        df = df.loc[df['RGNTI'].apply(lambda x: re.findall("\d+", str(x)) != [])]  # Отбираем строки с кодом
        df_test = pd.read_csv(file_path + "/test_ru.csv", sep='\t', encoding='cp1251', on_bad_lines='skip')
        df_test = df_test.loc[df_test['RGNTI'].apply(lambda x: re.findall("\d+", str(x)) != [])]

        # Извлечение кода первого уровня (target)
        df['target'] = df['RGNTI'].apply(lambda x:
                                         list(set([re.findall("\d+", el)[0]
                                                   for el in str(x).split('\\') if re.findall("\d+", el)])))
        df_test['target'] = df_test['RGNTI'].apply(lambda x:
                                                   list(set([re.findall("\d+", el)[0]
                                                             for el in str(x).split('\\') if re.findall("\d+", el)])))
        # Извлечение кода третьего уровня (target_3): предполагается, что код имеет вид "X.Y.Z"
        df['target_3'] = df['RGNTI'].apply(lambda x:
                                           list(set([re.findall("\d+\.\d+\.\d+", el)[0]
                                                     for el in str(x).split('\\') if re.findall("\d+\.\d+\.\d+", el)])))
        df_test['target_3'] = df_test['RGNTI'].apply(lambda x:
                                                     list(set([re.findall("\d+\.\d+\.\d+", el)[0]
                                                               for el in str(x).split('\\') if
                                                               re.findall("\d+\.\d+\.\d+", el)])))
        # Если требуется удалить редкие значения по первому уровню
        df_trunc = df.copy()
        df_test_trunc = df_test.copy()
        if number_of_delteted_values > 0:
            list_of_few_values = pd.Series(np.concatenate(df['target'].values)).value_counts()[
                                 :-number_of_delteted_values].index.to_list()
            df_trunc['target'] = df['target'].apply(lambda x: list(set(x) & set(list_of_few_values)))
            df_trunc = df_trunc[df_trunc['target'].apply(lambda x: x != [])]
            df_test_trunc["target"] = df_test['target'].apply(lambda x: list(set(x) & set(list_of_few_values)))
            df_test_trunc["target"] = df_test_trunc["target"].apply(lambda x: x if x != [] else ["no class"])

        # Отбираем уникальные значения target (первого уровня)
        unique_vals = np.unique(np.concatenate(df_trunc['target'].values))

        # Фильтрация кодов третьего уровня по минимальному числу элементов
        list_of_proper_values_target_3 = []
        list_of_inproper_values_target_3 = []
        print(f"Удаление элементов третьего уровня, количество которых меньше {minimal_number_of_elements_RGNTI3}")
        print(df_trunc.head())
        for target_val in tqdm(unique_vals):
            # Извлекаем коды третьего уровня, начинающиеся с target_val (например, для target_val="12" ищем "12.X.Y")
            needed_target3 = df_trunc['target_3'].apply(lambda x: [re.findall(f"{target_val}\.\d+\.\d+", el)[0]
                                                                   for el in x if
                                                                   re.findall(f"{target_val}\.\d+\.\d+", el)])
            # Объединяем списки и считаем вхождения
            try:
                concatenated_list_target3 = pd.Series(
                    np.concatenate(np.array([el for el in needed_target3.values.tolist() if el],
                                            dtype="object"))).value_counts()
            except ValueError:
                continue
            list_of_proper_values_target_3.extend(
                concatenated_list_target3[
                    concatenated_list_target3 >= minimal_number_of_elements_RGNTI3].index.to_list())
            list_of_inproper_values_target_3.extend(
                concatenated_list_target3[
                    concatenated_list_target3 < minimal_number_of_elements_RGNTI3].index.to_list())
        set_of_proper_values_target_3 = set(list_of_proper_values_target_3)

        # Создаём копию для дальнейшей обработки рубрик 3-го уровня
        df_trunc3 = df_trunc.copy()
        # График для удаляемых элементов (опционально)
        df_trunc3_deleted = df_trunc['target_3'].apply(lambda x: list(set(x) - set_of_proper_values_target_3))
        df_trunc3_deleted = pd.Series([el for el in df_trunc3_deleted if el != []])

        # Фильтруем, оставляя только те target_3, которые присутствуют в наборе proper
        df_trunc3['target_3'] = df_trunc['target_3'].apply(lambda x: list(set(x) & set_of_proper_values_target_3))
        df_trunc3 = df_trunc3[df_trunc3['target_3'].apply(lambda x: x != [])]

        if minimal_number_of_elements_RGNTI3 > 1:
            fig = plt.figure(facecolor="#fff3e0", figsize=(8, 18), dpi=500)
            deleted_values_count = pd.Series(np.concatenate(df_trunc3_deleted.values)).value_counts()
            deleted_values_count = pd.DataFrame({'RGNTI 3': deleted_values_count.index,
                                                 'Количество элементов': deleted_values_count.values})
            print(deleted_values_count.shape)
            sns_plot1 = sns.barplot(y="RGNTI 3", x="Количество элементов", data=deleted_values_count)
            sns_plot1.tick_params(labelsize=6)
            plt.xticks(fontsize=10)
            plt.tight_layout()
            plt.title("Количество удаляемых текстов из датасета для 3-его уровня ГРНТИ")
            if dir_name:
                plt.savefig(dir_name + "Количество удаляемых текстов из датасета для 3-его уровня ГРНТИ.png",
                            bbox_inches='tight')
        else:
            print("Элементы не удаляются из датасета по RGNTI 3")
        fig = plt.figure(facecolor="#fff3e0", figsize=(8, 18), dpi=500)
        residual_values = pd.Series(np.concatenate(df_trunc3['target_3'].values)).value_counts()
        residual_values = pd.DataFrame({'RGNTI 3': residual_values.index,
                                        'Количество элементов': residual_values.values})
        print(residual_values.shape)
        sns_plot2 = sns.barplot(y="RGNTI 3", x="Количество элементов", data=residual_values)
        sns_plot2.tick_params(labelsize=6)
        plt.xticks(fontsize=10)
        plt.tight_layout()
        plt.title("Количество элементов, остающихся в датасете для 3-его уровня ГРНТИ")
        if dir_name:
            plt.savefig(dir_name + "Количество элементов, остающихся в датасете для 3-его уровня ГРНТИ.png",
                        bbox_inches='tight')

        df_test_trunc3 = df_test_trunc.copy()

        # Кодирование классов первого уровня (target)
        union_of_targets = set(unique_vals)
        coding = range(len(union_of_targets))
        dict_Vinit_code_int = dict(zip(union_of_targets, coding))
        curr_path1 = os.path.join(grnti_folder, "my_grnti1_int.json")
        if change_codes:
            with open(curr_path1, "w") as outfile:
                json.dump(dict_Vinit_code_int, outfile)
        with open(curr_path1, "r") as code_file:
            grnti_mapping_dict = json.load(code_file)
        n_classes = len(grnti_mapping_dict)

        # Кодирование классов третьего уровня (target_3)
        unique_vals_level3 = np.unique(np.concatenate(df_trunc3['target_3'].values))
        union_of_targets3 = set(unique_vals_level3)
        coding3 = range(len(union_of_targets3))
        dict_Vinit_code_int3 = dict(zip(union_of_targets3, coding3))
        if change_codes:
            with open(grnti_folder + "my_grnti3_int.json", "w") as outfile:
                json.dump(dict_Vinit_code_int3, outfile)
        with open(grnti_folder + 'my_grnti3_int.json', "r") as code_file:
            grnti_mapping_dict3 = json.load(code_file)
        n_classes3 = len(grnti_mapping_dict3)

        # Кодирование классов для тренировочного датасета (target)
        df_trunc_result_multiclass_targets = []
        for list_el in df_trunc3['target']:
            classes_zero = [0] * n_classes
            for index in list_el:
                if index in grnti_mapping_dict.keys():
                    classes_zero[grnti_mapping_dict[index]] = 1
            df_trunc_result_multiclass_targets.append(classes_zero)
        # Кодирование классов для тестового датасета (target)
        df_test_trunc_result_multiclass_targets = []
        for list_el in df_test_trunc3['target']:
            classes_zero = [0] * n_classes
            for index in list_el:
                if index in grnti_mapping_dict.keys():
                    classes_zero[grnti_mapping_dict[index]] = 1
            df_test_trunc_result_multiclass_targets.append(classes_zero)
        df_trunc3['target_coded'] = df_trunc_result_multiclass_targets
        df_test_trunc3['target_coded'] = df_test_trunc_result_multiclass_targets

        # Кодирование классов для тренировочного датасета (target_3)
        df_trunc_result_multiclass_targets3 = []
        for list_el in df_trunc3['target_3']:
            classes_zero = [0] * n_classes3
            for index in list_el:
                if index in grnti_mapping_dict3.keys():
                    classes_zero[grnti_mapping_dict3[index]] = 1
            df_trunc_result_multiclass_targets3.append(classes_zero)
        # Кодирование классов для тестового датасета (target_3)
        df_test_trunc_result_multiclass_targets3 = []
        for list_el in df_test_trunc3['target_3']:
            classes_zero = [0] * n_classes3
            for index in list_el:
                if index in grnti_mapping_dict3.keys():
                    classes_zero[grnti_mapping_dict3[index]] = 1
            df_test_trunc_result_multiclass_targets3.append(classes_zero)
        df_trunc3['target_coded3'] = df_trunc_result_multiclass_targets3
        df_test_trunc3['target_coded3'] = df_test_trunc_result_multiclass_targets3

        ############################
        # Формирование итогового текстового поля
        df_trunc3['text'] = (df_trunc3['title'].apply(lambda x: x + ' [SEP] ')
                             + df_trunc3['ref_txt'])
        df_test_trunc3['text'] = (df_test_trunc3['title'].apply(lambda x: x + ' [SEP] ')
                                  + df_test_trunc3['ref_txt'])

        df_trunc3['text'] = (df_trunc3['text'].apply(lambda x: str(x) + ' [SEP] ') + df_trunc3['kw_list'])
        df_test_trunc3['text'] = df_test_trunc3['text'].apply(lambda x: str(x) + ' [SEP] ') + df_test_trunc3['kw_list']

        # Удаляем строки с пустым текстом и с количеством слов меньше минимума
        df_trunc3 = df_trunc3.dropna(subset=['text'], axis=0)
        df_test_trunc3 = df_test_trunc3.dropna(subset=['text'], axis=0)
        df_trunc3 = df_trunc3[df_trunc3['text'].apply(lambda x: len(x.split()) > minimal_number_of_words)]

        print("Доля оставшихся элементов в тренировочном датасете: ", df_trunc3.shape[0] / df.shape[0])

        return df_trunc3, df_test_trunc3, n_classes, n_classes3


    def get_grnti1_BERT_dataframes(self, file_path, number_of_delteted_values,
                                   minimal_number_of_elements_RGNTI2,
                                   minimal_number_of_words,
                                   dir_name=None,
                                   change_codes=False,
                                   grnti_folder=""):

        df = pd.read_csv(file_path + "/train_ru.csv", sep='\t', encoding='cp1251')
        df = df.loc[df['RGNTI'].apply(lambda x: re.findall("\d+", x) != [])]  # Пропускаем строки без класса
        df_test = pd.read_csv(file_path + "/test_ru.csv", sep='\t', encoding='cp1251',
                              on_bad_lines='skip')  # error_bad_lines
        df_test = df_test.loc[df_test['RGNTI'].apply(lambda x: re.findall("\d+", x) != [])]
        df['target'] = df['RGNTI'].apply(lambda x:
                                         list(set([re.findall("\d+", el)[0]
                                                   for el in x.split(
                                                 '\\')])))  # Для каждой строки извлекаем значения ГРНТИ 1 уровня

        df_test['target'] = df_test['RGNTI'].apply(lambda x:
                                                   list(set([re.findall("\d+", el)[0]
                                                             for el in x.split('\\')])))
        df['target_2'] = df['RGNTI'].apply(lambda x:
                                           list(set([re.findall("\d+.\d+", el)[0]
                                                     for el in x.split('\\')])))
        df_test['target_2'] = df_test['RGNTI'].apply(lambda x:
                                                     list(set([re.findall("\d+.\d+", el)[0]
                                                               for el in x.split('\\')])))

        pd.Series((np.concatenate(df['target'].values))).value_counts()[-number_of_delteted_values:].plot.bar()
        plt.xlabel("RGNTI 1")
        plt.ylabel("Количество элементов")
        plt.title("Количество удаляемых текстов из датасета для 1-ого уровня ГРНТИ")

        # plt.show()
        if dir_name:
            plt.savefig(dir_name + "Количество удаляемых текстов из датасета для 1-ого уровня ГРНТИ.png",
                        bbox_inches='tight')

        pd.Series(np.concatenate(df['target'].values)).value_counts()[:-number_of_delteted_values].plot.bar()
        plt.xlabel("RGNTI 1")
        plt.ylabel("Количество элементов")
        plt.title("Количество элементов, остающихся в датасете")
        # plt.show()
        if dir_name:
            plt.savefig(dir_name + "Количество элементов, остающихся в датасете.png",
                        bbox_inches='tight')

        list_of_few_values = pd.Series(np.concatenate(
            df['target'].values)).value_counts()[:-number_of_delteted_values].index.to_list()
        df_trunc = df.copy()
        df_trunc['target'] = df['target'].apply(lambda x: list(set(x) & set(list_of_few_values)))
        df_trunc = df_trunc[df_trunc['target'].apply(lambda x: x != [])]

        df_test_trunc = df_test.copy()
        df_test_trunc["target"] = df_test['target'].apply(lambda x: list(set(x) & set(list_of_few_values)))

        df_test_trunc["target"] = df_test_trunc["target"].apply(lambda x: x if x != [] else ["no class"])

        unique_vals = np.unique(np.concatenate(df_trunc['target'].values))

        list_of_proper_values_target_2 = []
        list_of_inproper_values_target_2 = []
        print(f"Удаление элементов второго уровня, количство которых меньше {minimal_number_of_elements_RGNTI2}")
        print(df_trunc.head())
        for target_2_val in tqdm(unique_vals):
            needed_taget2 = df_trunc['target_2'].apply(lambda x: [re.findall(f"{target_2_val}.\d+", el)[0] for el
                                                                  in x if re.findall(f"{target_2_val}.\d+", el)])

            concatenated_list_target2 = pd.Series(np.concatenate(np.array([el for el
                                                                           in needed_taget2.values.tolist() if el],
                                                                          dtype="object"))).value_counts()
            list_of_proper_values_target_2.extend(
                concatenated_list_target2[concatenated_list_target2 >= minimal_number_of_elements_RGNTI2]. \
                index.to_list())
            list_of_inproper_values_target_2.extend(
                concatenated_list_target2[concatenated_list_target2 < minimal_number_of_elements_RGNTI2]. \
                index.to_list())

        set_of_proper_values_target_2 = set(list_of_proper_values_target_2)
        df_trunc2 = df_trunc.copy()
        df_trunc2['target_2'] = df_trunc['target_2'].apply(lambda x: list(set(x) &
                                                                          set_of_proper_values_target_2))
        df_trunc2 = df_trunc2[df_trunc2['target_2'].apply(lambda x: x != [])]

        df_test_trunc2 = df_test_trunc

        union_of_targets = set(unique_vals)
        coding = range(len(union_of_targets))
        dict_Vinit_code_int = dict(zip(union_of_targets, coding))
        if change_codes:
            with open(grnti_folder + "my_grnti1_int.json", "w") as outfile:
                json.dump(dict_Vinit_code_int, outfile)

        with open(grnti_folder + 'my_grnti1_int.json', "r") as code_file:
            grnti_mapping_dict = json.load(code_file)  # Загружаем файл с кодами
        n_classes = len(grnti_mapping_dict)

        # Уровень 2
        unique_vals_level2 = np.unique(np.concatenate(df_trunc2['target_2'].values))
        union_of_targets2 = set(unique_vals_level2)
        coding2 = range(len(union_of_targets2))
        dict_Vinit_code_int2 = dict(zip(union_of_targets2, coding2))
        if change_codes:
            with open(grnti_folder + "my_grnti2_int.json", "w") as outfile:
                json.dump(dict_Vinit_code_int2, outfile)

        with open(grnti_folder + 'my_grnti2_int.json', "r") as code_file:
            grnti_mapping_dict2 = json.load(code_file)  # Загружаем файл с кодами
        n_classes2 = len(grnti_mapping_dict2)

        # Кодируем классы тренировочного датасета
        df_trunc_result_multiclass_targets = []
        for list_el in df_trunc2['target']:
            classes_zero = [0] * n_classes
            for index in list_el:
                if index in grnti_mapping_dict.keys():
                    classes_zero[grnti_mapping_dict[index]] = 1

            df_trunc_result_multiclass_targets.append(classes_zero)

        # Кодируем классы тестового датасета
        df_test_trunc_result_multiclass_targets = []
        for list_el in df_test_trunc2['target']:
            classes_zero = [0] * n_classes
            for index in list_el:
                if index in grnti_mapping_dict.keys():
                    classes_zero[grnti_mapping_dict[index]] = 1

            df_test_trunc_result_multiclass_targets.append(classes_zero)
        df_trunc2['target_coded'] = df_trunc_result_multiclass_targets
        df_test_trunc2['target_coded'] = df_test_trunc_result_multiclass_targets

        # Кодируем классы тренировочного датасета level2
        df_trunc_result_multiclass_targets2 = []
        for list_el in df_trunc2['target_2']:
            classes_zero = [0] * n_classes2
            for index in list_el:
                if index in grnti_mapping_dict2.keys():
                    classes_zero[grnti_mapping_dict2[index]] = 1

            df_trunc_result_multiclass_targets2.append(classes_zero)

        # Кодируем классы тестового датасета level2
        df_test_trunc_result_multiclass_targets2 = []
        for list_el in df_test_trunc2['target_2']:
            classes_zero = [0] * n_classes2
            for index in list_el:
                if index in grnti_mapping_dict2.keys():
                    classes_zero[grnti_mapping_dict2[index]] = 1

            df_test_trunc_result_multiclass_targets2.append(classes_zero)

        df_trunc2['target_coded2'] = df_trunc_result_multiclass_targets2
        df_test_trunc2['target_coded2'] = df_test_trunc_result_multiclass_targets2
        ############################
        df_trunc2['text'] = (df_trunc2['title'].apply(lambda x: x + ' [SEP] ')
                             + df_trunc2['ref_txt'])
        df_test_trunc2['text'] = (df_test_trunc2['title'].apply(lambda x: x + ' [SEP] ')
                                  + df_test_trunc2['ref_txt'])

        df_trunc2['text'] = (df_trunc2['text'].apply(lambda x: str(x) + ' [SEP] ') + df_trunc2['kw_list'])

        df_test_trunc2['text'] = df_test_trunc2['text'].apply(lambda x: str(x) +
                                                                        ' [SEP] ') + df_test_trunc2['kw_list']

        df_trunc2 = df_trunc2.dropna(subset=['text'], axis=0)
        df_test_trunc2 = df_test_trunc2.dropna(subset=['text'], axis=0)
        df_trunc2 = df_trunc2[df_trunc2['text'].apply(lambda x: len(x.split()) > minimal_number_of_words)]

        print("Доля оставшихся элементов в тренировочном датасете: ", df_trunc2.shape[0] / df.shape[0])

        return df_trunc2, df_test_trunc2, n_classes, n_classes2

    def get_grnti1_2_BERT_dataframes(self, file_path, number_of_delteted_values,
                                     minimal_number_of_elements_RGNTI2,
                                     minimal_number_of_words,
                                     dir_name=None,
                                     change_codes=False,
                                     grnti_folder=""):
        df = pd.read_csv(file_path + "/train_ru.csv", sep='\t', encoding='cp1251')
        df = df.loc[df['RGNTI'].apply(lambda x: re.findall("\d+", x) != [])]  # Пропускаем строки без класса
        df_test = pd.read_csv(file_path + "/test_ru.csv", sep='\t', encoding='cp1251',
                              on_bad_lines='skip')
        df_test = df_test.loc[df_test['RGNTI'].apply(lambda x: re.findall("\d+", x) != [])]
        df['target'] = df['RGNTI'].apply(lambda x:
                                         list(set([re.findall("\d+", el)[0]
                                                   for el in x.split(
                                                 '\\')])))  # Для каждой строки извлекаем значения ГРНТИ 1 уровня

        df_test['target'] = df_test['RGNTI'].apply(lambda x:
                                                   list(set([re.findall("\d+", el)[0]
                                                             for el in x.split('\\')])))
        df['target_2'] = df['RGNTI'].apply(lambda x:
                                           list(set([re.findall("\d+.\d+", el)[0]
                                                     for el in x.split('\\')])))
        df_test['target_2'] = df_test['RGNTI'].apply(lambda x:
                                                     list(set([re.findall("\d+.\d+", el)[0]
                                                               for el in x.split('\\')])))

        df_trunc = df.copy()

        df_test_trunc = df_test.copy()
        if number_of_delteted_values > 0:
            list_of_few_values = pd.Series(np.concatenate(
                df['target'].values)).value_counts()[:-number_of_delteted_values].index.to_list()
            df_trunc['target'] = df['target'].apply(lambda x: list(set(x) & set(list_of_few_values)))
            df_trunc = df_trunc[df_trunc['target'].apply(lambda x: x != [])]

            df_test_trunc["target"] = df_test['target'].apply(lambda x: list(set(x) & set(list_of_few_values)))

            df_test_trunc["target"] = df_test_trunc["target"].apply(lambda x: x if x != [] else ["no class"])

        unique_vals = np.unique(np.concatenate(df_trunc['target'].values))

        list_of_proper_values_target_2 = []
        list_of_inproper_values_target_2 = []
        print(f"Удаление элементов второго уровня, количство которых меньше {minimal_number_of_elements_RGNTI2}")
        print(df_trunc.head())
        for target_2_val in tqdm(unique_vals):
            needed_taget2 = df_trunc['target_2'].apply(lambda x: [re.findall(f"{target_2_val}.\d+", el)[0] for el
                                                                  in x if re.findall(f"{target_2_val}.\d+", el)])

            concatenated_list_target2 = pd.Series(np.concatenate(np.array([el for el
                                                                           in needed_taget2.values.tolist() if el],
                                                                          dtype="object"))).value_counts()
            list_of_proper_values_target_2.extend(
                concatenated_list_target2[concatenated_list_target2 >= minimal_number_of_elements_RGNTI2]. \
                index.to_list())
            list_of_inproper_values_target_2.extend(
                concatenated_list_target2[concatenated_list_target2 < minimal_number_of_elements_RGNTI2]. \
                index.to_list())

        # print(list_of_inproper_values_target_2)
        set_of_proper_values_target_2 = set(list_of_proper_values_target_2)
        # print(set_of_proper_values_target_2)
        df_trunc2 = df_trunc.copy()
        # Код для графика
        # print("df_trunc['target_2']:", df_trunc['target_2'])
        df_trunc2_deleted = df_trunc['target_2'].apply(lambda x: list(set(x) -
                                                                      set_of_proper_values_target_2))
        df_trunc2_deleted = pd.Series([el for el in df_trunc2_deleted if el != []])
        # print("df_trunc2_deleted:", df_trunc2_deleted)
        # Конец кода для графика

        df_trunc2['target_2'] = df_trunc['target_2'].apply(lambda x: list(set(x) &
                                                                          set_of_proper_values_target_2))
        df_trunc2 = df_trunc2[df_trunc2['target_2'].apply(lambda x: x != [])]

        # код для графика 2
        if minimal_number_of_elements_RGNTI2 > 1:
            fig = plt.figure(facecolor="#fff3e0", figsize=(8, 18), dpi=500)
            deleted_values_count = pd.Series(np.concatenate(df_trunc2_deleted.values)).value_counts()
            deleted_values_count = pd.DataFrame({'RGNTI 2': deleted_values_count.index,
                                                 'Количество элементов': deleted_values_count.values})

            print(deleted_values_count.shape)

            # deleted_values_count.plot.bar()

            sns_plot1 = sns.barplot(y="RGNTI 2", x="Количество элементов", data=deleted_values_count)

            # plt.xlabel("RGNTI 2")
            # plt.ylabel("Количество элементов")
            # plt.rc('xtick', labelsize=6)

            sns_plot1.tick_params(labelsize=6)
            # plt.xticks(rotation=90)
            plt.xticks(fontsize=10)

            # sns_plot1.set_yticklabels(sns_plot1.get_yticks(), size = 6)
            # sns_plot1.set_xticklabels(sns_plot1.get_xticks(), size = 10)
            plt.tight_layout()

            plt.title("Количество удаляемых текстов из датасета для 2-ого уровня ГРНТИ")
            if dir_name:
                plt.savefig(dir_name + "Количество удаляемых текстов из датасета для 2-ого уровня ГРНТИ.png",
                            bbox_inches='tight')

        else:
            print("Элементы не удаляются из датасета по RGNTI 2")
        fig = plt.figure(facecolor="#fff3e0", figsize=(8, 18), dpi=500)
        residual_values = pd.Series(np.concatenate(df_trunc2['target_2'].values)).value_counts()
        residual_values = pd.DataFrame({'RGNTI 2': residual_values.index,
                                        'Количество элементов': residual_values.values})
        print(residual_values.shape)

        # residual_values.plot.bar()
        sns_plot2 = sns.barplot(y="RGNTI 2", x="Количество элементов", data=residual_values)

        # sns_plot2.set_yticklabels(sns_plot2.get_yticks(), size = 6)
        # sns_plot2.set_xticklabels(sns_plot2.get_xticks(), size = 10)

        sns_plot2.tick_params(labelsize=6)
        # plt.xticks(rotation=90)
        plt.xticks(fontsize=10)

        # plt.xlabel("RGNTI 2")
        # plt.ylabel("Количество элементов")
        # plt.rc('xtick', labelsize=6)
        plt.tight_layout()
        plt.title("Количество элементов, остающихся в датасете для 2-ого уровня ГРНТИ")
        # plt.show()
        if dir_name:
            plt.savefig(dir_name + "Количество элементов, остающихся в датасете для 2-ого уровня ГРНТИ.png",
                        bbox_inches='tight')
        # конец кода для графика 2

        df_test_trunc2 = df_test_trunc

        union_of_targets = set(unique_vals)
        coding = range(len(union_of_targets))
        dict_Vinit_code_int = dict(zip(union_of_targets, coding))

        if change_codes:
            with open(grnti_folder + "my_grnti1_int.json", "w") as outfile:
                json.dump(dict_Vinit_code_int, outfile)

        with open(grnti_folder + 'my_grnti1_int.json', "r") as code_file:
            grnti_mapping_dict = json.load(code_file)  # Загружаем файл с кодами
        n_classes = len(grnti_mapping_dict)

        # Уровень 2
        unique_vals_level2 = np.unique(np.concatenate(df_trunc2['target_2'].values))
        union_of_targets2 = set(unique_vals_level2)
        coding2 = range(len(union_of_targets2))
        dict_Vinit_code_int2 = dict(zip(union_of_targets2, coding2))
        if change_codes:
            with open(grnti_folder + "my_grnti2_int.json", "w") as outfile:
                json.dump(dict_Vinit_code_int2, outfile)

        with open(grnti_folder + 'my_grnti2_int.json', "r") as code_file:
            grnti_mapping_dict2 = json.load(code_file)  # Загружаем файл с кодами
        n_classes2 = len(grnti_mapping_dict2)

        # Кодируем классы тренировочного датасета
        df_trunc_result_multiclass_targets = []
        for list_el in df_trunc2['target']:
            classes_zero = [0] * n_classes
            for index in list_el:
                if index in grnti_mapping_dict.keys():
                    classes_zero[grnti_mapping_dict[index]] = 1

            df_trunc_result_multiclass_targets.append(classes_zero)

        # Кодируем классы тестового датасета
        df_test_trunc_result_multiclass_targets = []
        for list_el in df_test_trunc2['target']:
            classes_zero = [0] * n_classes
            for index in list_el:
                if index in grnti_mapping_dict.keys():
                    classes_zero[grnti_mapping_dict[index]] = 1

            df_test_trunc_result_multiclass_targets.append(classes_zero)
        df_trunc2['target_coded'] = df_trunc_result_multiclass_targets
        df_test_trunc2['target_coded'] = df_test_trunc_result_multiclass_targets

        # Кодируем классы тренировочного датасета level2
        df_trunc_result_multiclass_targets2 = []
        for list_el in df_trunc2['target_2']:
            classes_zero = [0] * n_classes2
            for index in list_el:
                if index in grnti_mapping_dict2.keys():
                    classes_zero[grnti_mapping_dict2[index]] = 1

            df_trunc_result_multiclass_targets2.append(classes_zero)

        # Кодируем классы тестового датасета level2
        df_test_trunc_result_multiclass_targets2 = []
        for list_el in df_test_trunc2['target_2']:
            classes_zero = [0] * n_classes2
            for index in list_el:
                if index in grnti_mapping_dict2.keys():
                    classes_zero[grnti_mapping_dict2[index]] = 1

            df_test_trunc_result_multiclass_targets2.append(classes_zero)

        df_trunc2['target_coded2'] = df_trunc_result_multiclass_targets2
        df_test_trunc2['target_coded2'] = df_test_trunc_result_multiclass_targets2
        ############################
        df_trunc2['text'] = (df_trunc2['title'].apply(lambda x: x + ' [SEP] ')
                             + df_trunc2['ref_txt'])
        df_test_trunc2['text'] = (df_test_trunc2['title'].apply(lambda x: x + ' [SEP] ')
                                  + df_test_trunc2['ref_txt'])

        df_trunc2['text'] = (df_trunc2['text'].apply(lambda x: str(x) + ' [SEP] ') + df_trunc2['kw_list'])

        df_test_trunc2['text'] = df_test_trunc2['text'].apply(lambda x: str(x) +
                                                                        ' [SEP] ') + df_test_trunc2['kw_list']

        df_trunc2 = df_trunc2.dropna(subset=['text'], axis=0)
        df_test_trunc2 = df_test_trunc2.dropna(subset=['text'], axis=0)
        df_trunc2 = df_trunc2[df_trunc2['text'].apply(lambda x: len(x.split()) > minimal_number_of_words)]

        print("Доля оставшихся элементов в тренировочном датасете: ", df_trunc2.shape[0] / df.shape[0])

        return df_trunc2, df_test_trunc2, n_classes, n_classes2

    def get_encoded_dataset(self, dataset, tokenizer,
                            max_length):

        def data_preprocesing(row):
            # Токенизация
            return tokenizer(row['text'], truncation=True, max_length=max_length)

        tokenized_dataset = dataset.map(data_preprocesing, batched=True)

        tokenized_dataset.set_format("torch")
        return tokenized_dataset

    def prepair_test_dataset(self, df_test, level,
                             max_number_tokens=512,
                             pre_trained_model_name='DeepPavlov/rubert-base-cased'):

        tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name, do_lower_case=True)

        dataset_test = Dataset.from_pandas(df_test[["text", f'target_coded{level}']]. \
                                           rename(columns={f'target_coded{level}': "label"}))

        print("Подготовка тестовых данных:")

        dataset_test = self.get_encoded_dataset(dataset_test, tokenizer=tokenizer,
                                           max_length=max_number_tokens)

        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
        return dataset_test, tokenizer, collate_fn

    def prepair_datasets(self, df, df_test, n_classes, level,
                         max_number_tokens=512,
                         pre_trained_model_name='DeepPavlov/rubert-base-cased'):
        tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name, do_lower_case=True)

        # Ищем элемнты, c list из target_coded, который встречается только 1 раз
        list_of_keys_less_than_two = []
        for key, val in Counter(df[f'target_coded{level}'].apply(lambda x: str(x))).items():
            if val < 2:
                list_of_keys_less_than_two.append(key)
        # Отделяем элементы датасета c list из target_coded, который встречается только 1 раз
        df_trunc_single_example = df[df[f'target_coded{level}'].apply(lambda x: str(x)
                                                                                in list_of_keys_less_than_two)]
        df_trunc_no_less_than_two = df[df[f'target_coded{level}'].apply(lambda x: str(x)
                                                                                  not in list_of_keys_less_than_two)]
        # Создаем стратифицированную выборку для обучения и валидации
        train_df_0, valid_df = train_test_split(df_trunc_no_less_than_two,
                                                stratify=df_trunc_no_less_than_two[f'target_coded{level}'].apply(
                                                    lambda x: str(x)),
                                                test_size=0.2)
        # Добавляем в обучающую выборку элементы начального датасета c list из target_coded, который встречается только 1 раз
        train_df = pd.concat([train_df_0, df_trunc_single_example], ignore_index=True)

        number_of_rows = train_df[f'target_coded{level}'].shape[0]
        number_per_class_2 = np.array([train_df[f'target_coded{level}'].apply(lambda x: x[index]).sum()
                                       for index in range(n_classes)])

        # Cчитаем веса каждого классов
        weights_per_class = torch.tensor(number_of_rows / (number_per_class_2 * n_classes))
        print("Веса для кажого класса: ", weights_per_class)

        dataset_train = Dataset.from_pandas(train_df[["text", f'target_coded{level}']]. \
                                            rename(columns={f'target_coded{level}': "label"}))

        dataset_valid = Dataset.from_pandas(valid_df[["text", f'target_coded{level}']]. \
                                            rename(columns={f'target_coded{level}': "label"}))

        dataset_test = Dataset.from_pandas(df_test[["text", f'target_coded{level}']]. \
                                           rename(columns={f'target_coded{level}': "label"}))
        print("Подготовка тренировочных данных:")

        dataset_train = self.get_encoded_dataset(dataset_train, tokenizer=tokenizer,
                                            max_length=max_number_tokens)
        print("Подготовка валидационных данных:")

        dataset_valid = self.get_encoded_dataset(dataset_valid, tokenizer=tokenizer,
                                            max_length=max_number_tokens)
        print("Подготовка тестовых данных:")

        dataset_test = self.get_encoded_dataset(dataset_test, tokenizer=tokenizer,
                                           max_length=max_number_tokens)

        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
        return dataset_train, dataset_valid, dataset_test, tokenizer, collate_fn, weights_per_class
        # loss_fuction_for_multiclass_classification

    def prepair_model(self, n_classes,
                      pre_trained_model_name='DeepPavlov/rubert-base-cased',
                      r=16,
                      lora_alpha=32,
                      lora_dropout=0.05):
        model = AutoModelForSequenceClassification.from_pretrained(pre_trained_model_name,
                                                                   problem_type="multi_label_classification",
                                                                   num_labels=n_classes)
        print(model)

        for name, param in zip(model.state_dict().items(), model.parameters()):
            if name[0] in ["bert.pooler.dense.weight",
                           "bert.pooler.dense.bias",
                           "classifier.weight",
                           "classifier.bias"]:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # lora для модели
        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["query", "key"],
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            modules_to_save=['classifier', 'bert.pooler']  # ["classifier"]
        )
        model_peft = get_peft_model(model, config)
        model_peft.print_trainable_parameters()

        return model_peft  # вместо return model

    # Функция подсчета всех метрик при валидации
    def prepair_compute_metrics(self, n_classes):

        multilabel_auroc_micro = MultilabelAUROC(num_labels=n_classes,
                                                 average="micro", thresholds=5)
        multilabel_auroc_macro = MultilabelAUROC(num_labels=n_classes,
                                                 average="macro", thresholds=5)
        multilabel_auroc_weighted = MultilabelAUROC(num_labels=n_classes,
                                                    average="weighted", thresholds=5)

        multilabel_accuracy_micro = MultilabelAccuracy(num_labels=n_classes, average='micro')
        multilabel_accuracy_macro = MultilabelAccuracy(num_labels=n_classes, average='macro')
        multilabel_accuracy_weighted = MultilabelAccuracy(num_labels=n_classes, average='weighted')
        threshold_list = [0.5, 0.6, 0.7, 0.8, 0.9]

        multilabel_f1_score_micro_list = [MultilabelF1Score(num_labels=n_classes, average='micro',
                                                            threshold=threshold) for threshold in threshold_list]
        multilabel_f1_score_macro_list = [MultilabelF1Score(num_labels=n_classes, average='macro',
                                                            threshold=threshold) for threshold in threshold_list]
        multilabel_f1_score_weighted_list = [MultilabelF1Score(num_labels=n_classes, average='weighted',
                                                               threshold=threshold) for threshold in threshold_list]

        def compute_metrics(pred):
            labels = torch.tensor(pred.label_ids).int()
            preds = torch.sigmoid(torch.tensor(pred.predictions).float())  # Принимем сигмоду для получения вероятностей

            accuracy_micro = multilabel_accuracy_micro(preds, labels)
            accuracy_macro = multilabel_accuracy_macro(preds, labels)
            accuracy_weighted = multilabel_accuracy_weighted(preds, labels)

            f1_micro = multilabel_f1_score_micro_list[0](preds, labels)
            f1_macro = multilabel_f1_score_macro_list[0](preds, labels)
            f1_weighted = multilabel_f1_score_weighted_list[0](preds, labels)

            f1_micro_06 = multilabel_f1_score_micro_list[1](preds, labels)
            f1_macro_06 = multilabel_f1_score_macro_list[1](preds, labels)
            f1_weighted_06 = multilabel_f1_score_weighted_list[1](preds, labels)

            f1_micro_07 = multilabel_f1_score_micro_list[2](preds, labels)
            f1_macro_07 = multilabel_f1_score_macro_list[2](preds, labels)
            f1_weighted_07 = multilabel_f1_score_weighted_list[2](preds, labels)

            f1_micro_08 = multilabel_f1_score_micro_list[3](preds, labels)
            f1_macro_08 = multilabel_f1_score_macro_list[3](preds, labels)
            f1_weighted_08 = multilabel_f1_score_weighted_list[3](preds, labels)

            f1_micro_09 = multilabel_f1_score_micro_list[4](preds, labels)
            f1_macro_09 = multilabel_f1_score_macro_list[4](preds, labels)
            f1_weighted_09 = multilabel_f1_score_weighted_list[4](preds, labels)

            aucroc_micro = multilabel_auroc_micro(preds, labels)
            aucroc_macro = multilabel_auroc_macro(preds, labels)
            aucroc_weighted = multilabel_auroc_weighted(preds, labels)

            return {
                'accuracy_micro_0.5': accuracy_micro,
                'accuracy_macro_0.5': accuracy_macro,
                'accuracy_weighted_0.5': accuracy_weighted,
                'f1_micro_0.5': f1_micro,
                'f1_macro_0.5': f1_macro,
                'f1_weighted_0.5': f1_weighted,

                'f1_micro_0.6': f1_micro_06,
                'f1_macro_0.6': f1_macro_06,
                'f1_weighted_0.6': f1_weighted_06,

                'f1_micro_0.7': f1_micro_07,
                'f1_macro_0.7': f1_macro_07,
                'f1_weighted_0.7': f1_weighted_07,

                'f1_micro_0.8': f1_micro_08,
                'f1_macro_0.8': f1_macro_08,
                'f1_weighted_0.8': f1_weighted_08,

                'f1_micro_0.9': f1_micro_09,
                'f1_macro_0.8': f1_macro_09,
                'f1_weighted_0.8': f1_weighted_09,

                "aucroc_micro": aucroc_micro,
                "aucroc_macro": aucroc_macro,
                "aucroc_weighted": aucroc_weighted
            }

        return compute_metrics

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
        with open('../../Downloads/output.csv', mode='w', newline='', encoding='windows-1251') as file:
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