# Стандартная библиотека Python
import re
import json

# Сторонние библиотеки для обработки данных
import pandas as pd
import numpy as np

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns

# tqdm для прогресс-баров
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm2
# Активация прогресс-баров для pandas
tqdm2.pandas()


def get_grnti1_2_3_BERT_dataframes(file_path, 
                                   number_of_deleted_values, 
                                   minimal_number_of_elements_RGNTI2,
                                   minimal_number_of_elements_RGNTI3,
                                   minimal_number_of_words,
                                   dir_name=None, 
                                   change_codes=False,
                                   grnti_folder=""):
    """
    Функция для обработки и фильтрации данных ГРНТИ трех уровней и создания графиков распределения элементов.
    
    Parameters:
    -----------
    file_path : str
        Путь к файлам данных.
    number_of_deleted_values : int
        Количество удаляемых значений ГРНТИ 1-го уровня.
    minimal_number_of_elements_RGNTI2 : int
        Минимальное количество элементов для ГРНТИ 2-го уровня.
    minimal_number_of_elements_RGNTI3 : int
        Минимальное количество элементов для ГРНТИ 3-го уровня.
    minimal_number_of_words : int
        Минимальное количество слов в тексте.
    dir_name : str, optional
        Директория для сохранения графиков. Если None, графики не сохраняются.
    change_codes : bool, default=False
        Флаг для перезаписи файлов с кодами ГРНТИ.
    grnti_folder : str, default=""
        Папка для хранения файлов соответствия кодов ГРНТИ.
        
    Returns:
    --------
    tuple
        (df_trunc3, df_test_trunc3, n_classes, n_classes2, n_classes3)
    """
    # Загрузка и первичная обработка данных
    df = pd.read_csv(file_path + "/train_ru.csv", sep='\t', encoding='cp1251')
    df = df.loc[df['RGNTI'].apply(lambda x: re.findall(r"\d+", x) != [])]  # Пропускаем строки без класса
    
    df_test = pd.read_csv(file_path + "/test_ru.csv", sep='\t', encoding='cp1251', on_bad_lines='skip')
    df_test = df_test.loc[df_test['RGNTI'].apply(lambda x: re.findall(r"\d+", x) != [])]
    
    # Извлекаем значения ГРНТИ 1-го уровня
    df['target'] = df['RGNTI'].apply(lambda x: 
                                    list(set([re.findall(r"\d+", el)[0] 
                                            for el in x.split('\\')])))
    df_test['target'] = df_test['RGNTI'].apply(lambda x: 
                                    list(set([re.findall(r"\d+", el)[0] 
                                            for el in x.split('\\')])))
    
    # Извлекаем значения ГРНТИ 2-го уровня
    df['target_2'] = df['RGNTI'].apply(lambda x: 
                                    list(set([re.findall(r"\d+\.\d+", el)[0] 
                                            for el in x.split('\\') if re.findall(r"\d+\.\d+", el)])))
    df_test['target_2'] = df_test['RGNTI'].apply(lambda x: 
                                    list(set([re.findall(r"\d+\.\d+", el)[0] 
                                            for el in x.split('\\') if re.findall(r"\d+\.\d+", el)])))
    
    # Извлекаем значения ГРНТИ 3-го уровня
    df['target_3'] = df['RGNTI'].apply(lambda x: 
                                    list(set([re.findall(r"\d+\.\d+\.\d+", el)[0] 
                                            for el in x.split('\\') if re.findall(r"\d+\.\d+\.\d+", el)])))
    df_test['target_3'] = df_test['RGNTI'].apply(lambda x: 
                                    list(set([re.findall(r"\d+\.\d+\.\d+", el)[0] 
                                            for el in x.split('\\') if re.findall(r"\d+\.\d+\.\d+", el)])))
    
    # Копируем данные для дальнейшей обработки
    df_trunc = df.copy()
    df_test_trunc = df_test.copy()
    
    # Обработка ГРНТИ 1-го уровня
    if number_of_deleted_values > 0:
        list_of_few_values = pd.Series(np.concatenate(
            df['target'].values)).value_counts()[:-number_of_deleted_values].index.to_list()
        
        df_trunc['target'] = df['target'].apply(lambda x: list(set(x) & set(list_of_few_values)))
        df_trunc = df_trunc[df_trunc['target'].apply(lambda x: x != [])]
        
        df_test_trunc["target"] = df_test['target'].apply(lambda x: list(set(x) & set(list_of_few_values)))
        df_test_trunc["target"] = df_test_trunc["target"].apply(lambda x: x if x != [] else ["no class"])
    
    # Создаем график для 1-го уровня ГРНТИ
    if dir_name and number_of_deleted_values > 0:
        # График удаленных элементов 1-го уровня
        removed_values_level1 = pd.Series(np.concatenate(
            df['target'].values)).value_counts().tail(number_of_deleted_values)
        removed_values_level1 = pd.DataFrame({
            'RGNTI 1': removed_values_level1.index, 
            'Количество элементов': removed_values_level1.values
        })
        
        fig = plt.figure(facecolor="#fff3e0", figsize=(8, 10), dpi=500)
        sns_plot_removed_1 = sns.barplot(y="RGNTI 1", x="Количество элементов", data=removed_values_level1)
        sns_plot_removed_1.tick_params(labelsize=8)
        plt.xticks(fontsize=10)
        plt.tight_layout()
        plt.title("Количество удаляемых текстов из датасета для 1-ого уровня ГРНТИ")
        plt.savefig(dir_name + "Количество удаляемых текстов из датасета для 1-ого уровня ГРНТИ.png", 
                    bbox_inches='tight')
        plt.close()
        
        # График оставшихся элементов 1-го уровня
        remaining_values_level1 = pd.Series(np.concatenate(
            df_trunc['target'].values)).value_counts()
        remaining_values_level1 = pd.DataFrame({
            'RGNTI 1': remaining_values_level1.index, 
            'Количество элементов': remaining_values_level1.values
        })
        
        fig = plt.figure(facecolor="#fff3e0", figsize=(8, 14), dpi=500)
        sns_plot_remain_1 = sns.barplot(y="RGNTI 1", x="Количество элементов", data=remaining_values_level1)
        sns_plot_remain_1.tick_params(labelsize=8)
        plt.xticks(fontsize=10)
        plt.tight_layout()
        plt.title("Количество элементов, остающихся в датасете для 1-ого уровня ГРНТИ")
        plt.savefig(dir_name + "Количество элементов, остающихся в датасете для 1-ого уровня ГРНТИ.png", 
                    bbox_inches='tight')
        plt.close()
    
    # Получаем уникальные значения ГРНТИ 1-го уровня
    unique_vals = np.unique(np.concatenate(df_trunc['target'].values))
    
    # Обработка ГРНТИ 2-го уровня
    print(f"Удаление элементов второго уровня, количество которых меньше {minimal_number_of_elements_RGNTI2}")
    
    list_of_proper_values_target_2 = []
    list_of_inproper_values_target_2 = []
    
    for target_2_val in tqdm(unique_vals):
        needed_target2 = df_trunc['target_2'].apply(lambda x: [re.findall(f"{target_2_val}\.\\d+", el)[0] for el 
                                                   in x if re.findall(f"{target_2_val}\.\\d+", el)])
        
        non_empty_lists = [el for el in needed_target2.values.tolist() if el]
        if non_empty_lists:
            concatenated_list_target2 = pd.Series(np.concatenate(non_empty_lists)).value_counts()
            list_of_proper_values_target_2.extend(
                concatenated_list_target2[concatenated_list_target2 >= minimal_number_of_elements_RGNTI2].index.to_list()
            )
            list_of_inproper_values_target_2.extend(
                concatenated_list_target2[concatenated_list_target2 < minimal_number_of_elements_RGNTI2].index.to_list()
            )
    
    set_of_proper_values_target_2 = set(list_of_proper_values_target_2)
    df_trunc2 = df_trunc.copy()
    
    df_trunc2_deleted = df_trunc['target_2'].apply(lambda x: list(set(x) - set_of_proper_values_target_2))
    df_trunc2_deleted = pd.Series([el for el in df_trunc2_deleted if el != []])
    
    df_trunc2['target_2'] = df_trunc['target_2'].apply(lambda x: list(set(x) & set_of_proper_values_target_2))
    df_trunc2 = df_trunc2[df_trunc2['target_2'].apply(lambda x: x != [])]
    
    # Создаем график для 2-го уровня ГРНТИ
    if dir_name and minimal_number_of_elements_RGNTI2 > 1:
        # График удаленных элементов 2-го уровня
        deleted_values_level2 = pd.Series(np.concatenate(df_trunc2_deleted.values)).value_counts()
        deleted_values_level2 = pd.DataFrame({
            'RGNTI 2': deleted_values_level2.index, 
            'Количество элементов': deleted_values_level2.values
        })
        
        fig = plt.figure(facecolor="#fff3e0", figsize=(8, 16), dpi=500)
        sns_plot_removed_2 = sns.barplot(y="RGNTI 2", x="Количество элементов", data=deleted_values_level2)
        sns_plot_removed_2.tick_params(labelsize=6)
        plt.xticks(fontsize=10)
        plt.tight_layout()
        plt.title("Количество удаляемых текстов из датасета для 2-ого уровня ГРНТИ")
        plt.savefig(dir_name + "Количество удаляемых текстов из датасета для 2-ого уровня ГРНТИ.png", 
                    bbox_inches='tight')
        plt.close()
    
    if dir_name:
        # График оставшихся элементов 2-го уровня
        remaining_values_level2 = pd.Series(np.concatenate(df_trunc2['target_2'].values)).value_counts()
        remaining_values_level2 = pd.DataFrame({
            'RGNTI 2': remaining_values_level2.index, 
            'Количество элементов': remaining_values_level2.values
        })
        
        fig = plt.figure(facecolor="#fff3e0", figsize=(8, 16), dpi=500)
        sns_plot_remain_2 = sns.barplot(y="RGNTI 2", x="Количество элементов", data=remaining_values_level2)
        sns_plot_remain_2.tick_params(labelsize=6)
        plt.xticks(fontsize=10)
        plt.tight_layout()
        plt.title("Количество элементов, остающихся в датасете для 2-ого уровня ГРНТИ")
        plt.savefig(dir_name + "Количество элементов, остающихся в датасете для 2-ого уровня ГРНТИ.png", 
                    bbox_inches='tight')
        plt.close()
    
    # Обработка тестового датасета для 2-го уровня
    df_test_trunc2 = df_test_trunc.copy()
    
    # Обработка ГРНТИ 3-го уровня
    print(f"Удаление элементов третьего уровня, количество которых меньше {minimal_number_of_elements_RGNTI3}")
    
    list_of_proper_values_target_3 = []
    list_of_inproper_values_target_3 = []
    
    for target_2_val in tqdm(set_of_proper_values_target_2):
        needed_target3 = df_trunc2['target_3'].apply(lambda x: [re.findall(f"{target_2_val}\.\\d+", el)[0] for el 
                                                   in x if re.findall(f"{target_2_val}\.\\d+", el)])
        
        non_empty_lists = [el for el in needed_target3.values.tolist() if el]
        if non_empty_lists:
            concatenated_list_target3 = pd.Series(np.concatenate(non_empty_lists)).value_counts()
            list_of_proper_values_target_3.extend(
                concatenated_list_target3[concatenated_list_target3 >= minimal_number_of_elements_RGNTI3].index.to_list()
            )
            list_of_inproper_values_target_3.extend(
                concatenated_list_target3[concatenated_list_target3 < minimal_number_of_elements_RGNTI3].index.to_list()
            )
    
    set_of_proper_values_target_3 = set(list_of_proper_values_target_3)
    df_trunc3 = df_trunc2.copy()
    
    df_trunc3_deleted = df_trunc2['target_3'].apply(lambda x: list(set(x) - set_of_proper_values_target_3))
    df_trunc3_deleted = pd.Series([el for el in df_trunc3_deleted if el != []])
    
    df_trunc3['target_3'] = df_trunc2['target_3'].apply(lambda x: list(set(x) & set_of_proper_values_target_3))
    df_trunc3 = df_trunc3[df_trunc3['target_3'].apply(lambda x: x != [])]
    
    # Создаем график для 3-го уровня ГРНТИ
    if dir_name and minimal_number_of_elements_RGNTI3 > 1:
        # График удаленных элементов 3-го уровня
        deleted_values_level3 = pd.Series(np.concatenate(df_trunc3_deleted.values)).value_counts()
        deleted_values_level3 = pd.DataFrame({
            'RGNTI 3': deleted_values_level3.index, 
            'Количество элементов': deleted_values_level3.values
        })
        
        fig = plt.figure(facecolor="#fff3e0", figsize=(8, 18), dpi=500)
        sns_plot_removed_3 = sns.barplot(y="RGNTI 3", x="Количество элементов", data=deleted_values_level3)
        sns_plot_removed_3.tick_params(labelsize=6)
        plt.xticks(fontsize=10)
        plt.tight_layout()
        plt.title("Количество удаляемых текстов из датасета для 3-ого уровня ГРНТИ")
        plt.savefig(dir_name + "Количество удаляемых текстов из датасета для 3-ого уровня ГРНТИ.png", 
                    bbox_inches='tight')
        plt.close()
    
    if dir_name:
        # График оставшихся элементов 3-го уровня
        remaining_values_level3 = pd.Series(np.concatenate(df_trunc3['target_3'].values)).value_counts()
        remaining_values_level3 = pd.DataFrame({
            'RGNTI 3': remaining_values_level3.index, 
            'Количество элементов': remaining_values_level3.values
        })
        
        fig = plt.figure(facecolor="#fff3e0", figsize=(8, 18), dpi=500)
        sns_plot_remain_3 = sns.barplot(y="RGNTI 3", x="Количество элементов", data=remaining_values_level3)
        sns_plot_remain_3.tick_params(labelsize=6)
        plt.xticks(fontsize=10)
        plt.tight_layout()
        plt.title("Количество элементов, остающихся в датасете для 3-ого уровня ГРНТИ")
        plt.savefig(dir_name + "Количество элементов, остающихся в датасете для 3-ого уровня ГРНТИ.png", 
                    bbox_inches='tight')
        plt.close()
    
    df_test_trunc3 = df_test_trunc2
    
    # Создание кодирования для ГРНТИ 1-го уровня
    union_of_targets = set(unique_vals)
    coding = range(len(union_of_targets))
    dict_Vinit_code_int = dict(zip(union_of_targets, coding))
    
    if change_codes:
        with open(grnti_folder + "my_grnti1_int.json", "w") as outfile:
            json.dump(dict_Vinit_code_int, outfile)
    
    with open(grnti_folder + 'my_grnti1_int.json', "r") as code_file:
        grnti_mapping_dict = json.load(code_file)  # Загружаем файл с кодами
    n_classes = len(grnti_mapping_dict)
    
    # Создание кодирования для ГРНТИ 2-го уровня
    unique_vals_level2 = np.unique(np.concatenate(df_trunc3['target_2'].values))
    union_of_targets2 = set(unique_vals_level2)
    coding2 = range(len(union_of_targets2))
    dict_Vinit_code_int2 = dict(zip(union_of_targets2, coding2))
    
    if change_codes:
        with open(grnti_folder + "my_grnti2_int.json", "w") as outfile:
            json.dump(dict_Vinit_code_int2, outfile)
    
    with open(grnti_folder + 'my_grnti2_int.json', "r") as code_file:
        grnti_mapping_dict2 = json.load(code_file)  # Загружаем файл с кодами
    n_classes2 = len(grnti_mapping_dict2)
    
    # Создание кодирования для ГРНТИ 3-го уровня
    unique_vals_level3 = np.unique(np.concatenate(df_trunc3['target_3'].values))
    union_of_targets3 = set(unique_vals_level3)
    coding3 = range(len(union_of_targets3))
    dict_Vinit_code_int3 = dict(zip(union_of_targets3, coding3))
    
    if change_codes:
        with open(grnti_folder + "my_grnti3_int.json", "w") as outfile:
            json.dump(dict_Vinit_code_int3, outfile)
    
    with open(grnti_folder + 'my_grnti3_int.json', "r") as code_file:
        grnti_mapping_dict3 = json.load(code_file)  # Загружаем файл с кодами
    n_classes3 = len(grnti_mapping_dict3)
    
    # Кодирование классов для 1-го уровня
    def encode_targets(target_lists, mapping_dict, n_classes):
        """Вспомогательная функция для кодирования целевых меток."""
        result = []
        for list_el in target_lists:
            classes_zero = [0] * n_classes
            for index in list_el:
                if index in mapping_dict.keys():
                    classes_zero[mapping_dict[index]] = 1
            result.append(classes_zero)
        return result
    
    # Кодируем классы ГРНТИ всех уровней
    df_trunc3['target_coded'] = encode_targets(df_trunc3['target'], grnti_mapping_dict, n_classes)
    df_test_trunc3['target_coded'] = encode_targets(df_test_trunc3['target'], grnti_mapping_dict, n_classes)
    
    df_trunc3['target_coded2'] = encode_targets(df_trunc3['target_2'], grnti_mapping_dict2, n_classes2)
    df_test_trunc3['target_coded2'] = encode_targets(df_test_trunc3['target_2'], grnti_mapping_dict2, n_classes2)
    
    df_trunc3['target_coded3'] = encode_targets(df_trunc3['target_3'], grnti_mapping_dict3, n_classes3)
    df_test_trunc3['target_coded3'] = encode_targets(df_test_trunc3['target_3'], grnti_mapping_dict3, n_classes3)
    
    # Формирование текстовых данных
    df_trunc3['text'] = (df_trunc3['title'].apply(lambda x: str(x) + ' [SEP] ') + 
                         df_trunc3['ref_txt'].apply(lambda x: str(x)) + 
                         ' [SEP] ' + df_trunc3['kw_list'].apply(lambda x: str(x)))
    
    df_test_trunc3['text'] = (df_test_trunc3['title'].apply(lambda x: str(x) + ' [SEP] ') + 
                              df_test_trunc3['ref_txt'].apply(lambda x: str(x)) + 
                              ' [SEP] ' + df_test_trunc3['kw_list'].apply(lambda x: str(x)))
    
    # Финальная фильтрация данных
    df_trunc3 = df_trunc3.dropna(subset=['text'], axis=0)
    df_test_trunc3 = df_test_trunc3.dropna(subset=['text'], axis=0)
    df_trunc3 = df_trunc3[df_trunc3['text'].apply(lambda x: len(str(x).split()) > minimal_number_of_words)]
    
    print("Доля оставшихся элементов в тренировочном датасете: ", df_trunc3.shape[0] / df.shape[0])
    
    return df_trunc3, df_test_trunc3, n_classes, n_classes2, n_classes3
