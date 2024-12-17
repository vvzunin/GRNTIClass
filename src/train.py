import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm2
import json
import torch
from peft import TaskType, LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification,\
    AutoTokenizer, DataCollatorWithPadding, Trainer
from collections import Counter
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MultilabelF1Score, MultilabelAccuracy, MultilabelAUROC, MultilabelPrecision, MultilabelRecall
from datasets import Dataset
from TrainSettings import TrainSettings

from ignite.metrics import ClassificationReport
from ignite.engine.engine import Engine

from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall
from ignite.metrics.metrics_lambda import MetricsLambda


tqdm2.pandas()
def get_grnti1_BERT_dataframes(file_path, number_of_delteted_values, 
                                 minimal_number_of_elements_RGNTI2,
                                 minimal_number_of_words,
                                 dir_name=None, 
                                 change_codes=False,
                                 grnti_folder = ""):
    
    df = pd.read_csv(file_path + "/train_ru.csv", sep='\t', encoding='cp1251')
    df = df.loc[df['RGNTI'].apply(lambda x: re.findall("\d+",x)!=[])] # Пропускаем строки без класса
    df_test = pd.read_csv(file_path + "/test_ru.csv", sep='\t', encoding='cp1251', on_bad_lines='skip')#error_bad_lines
    df_test = df_test.loc[df_test['RGNTI'].apply(lambda x: re.findall("\d+",x)!=[])]
    df['target'] = df['RGNTI'].apply(lambda x:
                                    list(set([re.findall("\d+",el)[0]
                                                for el in x.split('\\')]))) # Для каждой строки извлекаем значения ГРНТИ 1 уровня

    df_test['target'] = df_test['RGNTI'].apply(lambda x:
                                    list(set([re.findall("\d+",el)[0]
                                                for el in x.split('\\')])))
    df['target_2'] = df['RGNTI'].apply(lambda x:
                                    list(set([re.findall("\d+.\d+",el)[0]
                                                for el in x.split('\\')])))
    df_test['target_2'] = df_test['RGNTI'].apply(lambda x:
                                    list(set([re.findall("\d+.\d+",el)[0]
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
    df_trunc = df_trunc[df_trunc['target'].apply(lambda x: x != []) ]

    df_test_trunc = df_test.copy()
    df_test_trunc["target"] = df_test['target'].apply(lambda x: list(set(x) & set(list_of_few_values)))

    df_test_trunc["target"] = df_test_trunc["target"].apply(lambda x: x if x!=[] else ["no class"])


    unique_vals = np.unique(np.concatenate(df_trunc['target'].values))

    list_of_proper_values_target_2 = []
    list_of_inproper_values_target_2 = []
    print(f"Удаление элементов второго уровня, количство которых меньше {minimal_number_of_elements_RGNTI2}")
    print(df_trunc.head())
    for target_2_val in tqdm(unique_vals):
        needed_taget2 = df_trunc['target_2'].apply(lambda x: [re.findall(f"{target_2_val}.\d+",el)[0] for el 
                                                    in x if re.findall(f"{target_2_val}.\d+",el)])
        
        concatenated_list_target2 = pd.Series(np.concatenate(np.array([el for el
                                                    in needed_taget2.values.tolist() if el], dtype="object"))).value_counts()
        list_of_proper_values_target_2.extend(concatenated_list_target2[concatenated_list_target2 >= minimal_number_of_elements_RGNTI2].\
                                            index.to_list())
        list_of_inproper_values_target_2.extend(concatenated_list_target2[concatenated_list_target2 < minimal_number_of_elements_RGNTI2].\
                                            index.to_list())
        

    set_of_proper_values_target_2 = set(list_of_proper_values_target_2)
    df_trunc2 = df_trunc.copy()
    df_trunc2['target_2'] = df_trunc['target_2'].apply(lambda x: list(set(x) &
                                                        set_of_proper_values_target_2))
    df_trunc2 = df_trunc2[df_trunc2['target_2'].apply(lambda x: x != []) ]

    df_test_trunc2 = df_test_trunc

    union_of_targets = set(unique_vals)
    coding =  range(len(union_of_targets))
    dict_Vinit_code_int = dict(zip(union_of_targets, coding))
    if change_codes:
        with open(grnti_folder + "my_grnti1_int.json", "w") as outfile:
            json.dump(dict_Vinit_code_int, outfile)


    with open(grnti_folder + 'my_grnti1_int.json', "r") as code_file:
        grnti_mapping_dict = json.load(code_file) # Загружаем файл с кодами 
    n_classes = len(grnti_mapping_dict)

    #Уровень 2
    unique_vals_level2 = np.unique(np.concatenate(df_trunc2['target_2'].values))
    union_of_targets2 = set(unique_vals_level2)
    coding2 =  range(len(union_of_targets2))
    dict_Vinit_code_int2 = dict(zip(union_of_targets2, coding2))
    if change_codes:
        with open(grnti_folder + "my_grnti2_int.json", "w") as outfile:
            json.dump(dict_Vinit_code_int2, outfile)


    with open(grnti_folder + 'my_grnti2_int.json', "r") as code_file:
        grnti_mapping_dict2 = json.load(code_file) # Загружаем файл с кодами 
    n_classes2 = len(grnti_mapping_dict2)

    #Кодируем классы тренировочного датасета
    df_trunc_result_multiclass_targets = []
    for list_el in df_trunc2['target']:
        classes_zero = [0] * n_classes
        for index in list_el:
            if index in grnti_mapping_dict.keys():
                classes_zero[grnti_mapping_dict[index]] = 1

        df_trunc_result_multiclass_targets.append(classes_zero)

    #Кодируем классы тестового датасета
    df_test_trunc_result_multiclass_targets = []
    for list_el in df_test_trunc2['target']:
        classes_zero = [0] * n_classes
        for index in list_el:
            if index in grnti_mapping_dict.keys():
                classes_zero[grnti_mapping_dict[index]] = 1

        df_test_trunc_result_multiclass_targets.append(classes_zero)
    df_trunc2['target_coded'] = df_trunc_result_multiclass_targets
    df_test_trunc2['target_coded'] = df_test_trunc_result_multiclass_targets

    #Кодируем классы тренировочного датасета level2
    df_trunc_result_multiclass_targets2 = []
    for list_el in df_trunc2['target_2']:
        classes_zero = [0] * n_classes2
        for index in list_el:
            if index in grnti_mapping_dict2.keys():
                classes_zero[grnti_mapping_dict2[index]] = 1

        df_trunc_result_multiclass_targets2.append(classes_zero)

   #Кодируем классы тестового датасета level2
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
    df_trunc2['text'] = (df_trunc2['title'].apply(lambda x:x+' [SEP] ') 
                     + df_trunc2['ref_txt'])
    df_test_trunc2['text'] = (df_test_trunc2['title'].apply(lambda x:x+' [SEP] ')
                                                + df_test_trunc2['ref_txt'])
    
    df_trunc2['text'] = (df_trunc2['text'].apply(lambda x:str(x)+' [SEP] ' ) + df_trunc2['kw_list'])

    df_test_trunc2['text'] = df_test_trunc2['text'].apply(lambda x:str(x)+
                                                          ' [SEP] ') + df_test_trunc2['kw_list']
    
    df_trunc2 = df_trunc2.dropna(subset=['text'], axis=0)
    df_test_trunc2 = df_test_trunc2.dropna(subset=['text'], axis=0)
    df_trunc2 = df_trunc2[df_trunc2['text'].apply(lambda x: len(x.split()) > minimal_number_of_words)]

    print("Доля оставшихся элементов в тренировочном датасете: ", df_trunc2.shape[0] / df.shape[0])

    return df_trunc2, df_test_trunc2, n_classes, n_classes2



def get_grnti1_2_BERT_dataframes(file_path, number_of_delteted_values, 
                                 minimal_number_of_elements_RGNTI2,
                                 minimal_number_of_words,
                                 dir_name=None, 
                                 change_codes=False,
                                 grnti_folder = ""):
    df = pd.read_csv(file_path + "/train_ru.csv", sep='\t', encoding='cp1251')
    df = df.loc[df['RGNTI'].apply(lambda x: re.findall("\d+",x)!=[])] # Пропускаем строки без класса
    df_test = pd.read_csv(file_path + "/test_ru.csv", sep='\t', encoding='cp1251',
                        on_bad_lines='skip')
    df_test = df_test.loc[df_test['RGNTI'].apply(lambda x: re.findall("\d+",x)!=[])]
    df['target'] = df['RGNTI'].apply(lambda x:
                                    list(set([re.findall("\d+",el)[0]
                                                for el in x.split('\\')]))) # Для каждой строки извлекаем значения ГРНТИ 1 уровня

    df_test['target'] = df_test['RGNTI'].apply(lambda x:
                                    list(set([re.findall("\d+",el)[0]
                                                for el in x.split('\\')])))
    df['target_2'] = df['RGNTI'].apply(lambda x:
                                    list(set([re.findall("\d+.\d+",el)[0]
                                                for el in x.split('\\')])))
    df_test['target_2'] = df_test['RGNTI'].apply(lambda x:
                                    list(set([re.findall("\d+.\d+",el)[0]
                                                for el in x.split('\\')])))


    df_trunc = df.copy()

    df_test_trunc = df_test.copy()
    if number_of_delteted_values > 0:

        list_of_few_values = pd.Series(np.concatenate(
            df['target'].values)).value_counts()[:-number_of_delteted_values].index.to_list() 
        df_trunc['target'] = df['target'].apply(lambda x: list(set(x) & set(list_of_few_values)))
        df_trunc = df_trunc[df_trunc['target'].apply(lambda x: x != []) ]

        df_test_trunc["target"] = df_test['target'].apply(lambda x: list(set(x) & set(list_of_few_values)))

        df_test_trunc["target"] = df_test_trunc["target"].apply(lambda x: x if x!=[] else ["no class"])


    unique_vals = np.unique(np.concatenate(df_trunc['target'].values))

    list_of_proper_values_target_2 = []
    list_of_inproper_values_target_2 = []
    print(f"Удаление элементов второго уровня, количство которых меньше {minimal_number_of_elements_RGNTI2}")
    print(df_trunc.head())
    for target_2_val in tqdm(unique_vals):
        needed_taget2 = df_trunc['target_2'].apply(lambda x: [re.findall(f"{target_2_val}.\d+",el)[0] for el 
                                                    in x if re.findall(f"{target_2_val}.\d+",el)])
        
        concatenated_list_target2 = pd.Series(np.concatenate(np.array([el for el
                                                    in needed_taget2.values.tolist() if el], dtype="object"))).value_counts()
        list_of_proper_values_target_2.extend(concatenated_list_target2[concatenated_list_target2 >= minimal_number_of_elements_RGNTI2].\
                                            index.to_list())
        list_of_inproper_values_target_2.extend(concatenated_list_target2[concatenated_list_target2 < minimal_number_of_elements_RGNTI2].\
                                            index.to_list())
        
    # print(list_of_inproper_values_target_2)
    set_of_proper_values_target_2 = set(list_of_proper_values_target_2)
    # print(set_of_proper_values_target_2)
    df_trunc2 = df_trunc.copy()
    #Код для графика 
    # print("df_trunc['target_2']:", df_trunc['target_2'])
    df_trunc2_deleted = df_trunc['target_2'].apply(lambda x: list(set(x) -
                                                    set_of_proper_values_target_2))
    df_trunc2_deleted = pd.Series([el for el in df_trunc2_deleted if el != []])
    # print("df_trunc2_deleted:", df_trunc2_deleted)
    #Конец кода для графика



    df_trunc2['target_2'] = df_trunc['target_2'].apply(lambda x: list(set(x) &
                                                        set_of_proper_values_target_2))
    df_trunc2 = df_trunc2[df_trunc2['target_2'].apply(lambda x: x != []) ]

    #код для графика 2
    if minimal_number_of_elements_RGNTI2 > 1:
        pd.Series(np.concatenate(df_trunc2_deleted.values)).value_counts().plot.bar()
        plt.xlabel("RGNTI 2")
        plt.ylabel("Количество элементов")
        plt.title("Количество удаляемых текстов из датасета для 2-ого уровня ГРНТИ")
        if dir_name:
            plt.savefig(dir_name + "Количество удаляемых текстов из датасета для 2-ого уровня ГРНТИ.png",
                        bbox_inches='tight')

    else:
        print("Элементы не удаляются из датасета по RGNTI 2")
 
    pd.Series(np.concatenate(df_trunc2['target_2'].values)).value_counts().plot.bar()
    plt.xlabel("RGNTI 2")
    plt.ylabel("Количество элементов")
    plt.title("Количество элементов, остающихся в датасете для 2-ого уровня ГРНТИ")
    # plt.show()
    if dir_name:
        plt.savefig(dir_name + "Количество элементов, остающихся в датасете для 2-ого уровня ГРНТИ.png",
                    bbox_inches='tight')
    #конец кода для графика 2

    df_test_trunc2 = df_test_trunc

    union_of_targets = set(unique_vals)
    coding =  range(len(union_of_targets))
    dict_Vinit_code_int = dict(zip(union_of_targets, coding))

    if change_codes:
        with open(grnti_folder + "my_grnti1_int.json", "w") as outfile:
            json.dump(dict_Vinit_code_int, outfile)


    with open(grnti_folder + 'my_grnti1_int.json', "r") as code_file:
        grnti_mapping_dict = json.load(code_file) # Загружаем файл с кодами 
    n_classes = len(grnti_mapping_dict)

    #Уровень 2
    unique_vals_level2 = np.unique(np.concatenate(df_trunc2['target_2'].values))
    union_of_targets2 = set(unique_vals_level2)
    coding2 =  range(len(union_of_targets2))
    dict_Vinit_code_int2 = dict(zip(union_of_targets2, coding2))
    if change_codes:
        with open(grnti_folder + "my_grnti2_int.json", "w") as outfile:
            json.dump(dict_Vinit_code_int2, outfile)


    with open(grnti_folder + 'my_grnti2_int.json', "r") as code_file:
        grnti_mapping_dict2 = json.load(code_file) # Загружаем файл с кодами 
    n_classes2 = len(grnti_mapping_dict2)

    #Кодируем классы тренировочного датасета
    df_trunc_result_multiclass_targets = []
    for list_el in df_trunc2['target']:
        classes_zero = [0] * n_classes
        for index in list_el:
            if index in grnti_mapping_dict.keys():
                classes_zero[grnti_mapping_dict[index]] = 1

        df_trunc_result_multiclass_targets.append(classes_zero)

    #Кодируем классы тестового датасета
    df_test_trunc_result_multiclass_targets = []
    for list_el in df_test_trunc2['target']:
        classes_zero = [0] * n_classes
        for index in list_el:
            if index in grnti_mapping_dict.keys():
                classes_zero[grnti_mapping_dict[index]] = 1

        df_test_trunc_result_multiclass_targets.append(classes_zero)
    df_trunc2['target_coded'] = df_trunc_result_multiclass_targets
    df_test_trunc2['target_coded'] = df_test_trunc_result_multiclass_targets

    #Кодируем классы тренировочного датасета level2
    df_trunc_result_multiclass_targets2 = []
    for list_el in df_trunc2['target_2']:
        classes_zero = [0] * n_classes2
        for index in list_el:
            if index in grnti_mapping_dict2.keys():
                classes_zero[grnti_mapping_dict2[index]] = 1

        df_trunc_result_multiclass_targets2.append(classes_zero)

   #Кодируем классы тестового датасета level2
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
    df_trunc2['text'] = (df_trunc2['title'].apply(lambda x:x+' [SEP] ') 
                     + df_trunc2['ref_txt'])
    df_test_trunc2['text'] = (df_test_trunc2['title'].apply(lambda x:x+' [SEP] ')
                                                + df_test_trunc2['ref_txt'])
    
    df_trunc2['text'] = (df_trunc2['text'].apply(lambda x:str(x)+' [SEP] ' ) + df_trunc2['kw_list'])

    df_test_trunc2['text'] = df_test_trunc2['text'].apply(lambda x:str(x)+
                                                          ' [SEP] ') + df_test_trunc2['kw_list']
    
    df_trunc2 = df_trunc2.dropna(subset=['text'], axis=0)
    df_test_trunc2 = df_test_trunc2.dropna(subset=['text'], axis=0)
    df_trunc2 = df_trunc2[df_trunc2['text'].apply(lambda x: len(x.split()) > minimal_number_of_words)]

    print("Доля оставшихся элементов в тренировочном датасете: ", df_trunc2.shape[0] / df.shape[0])

    return df_trunc2, df_test_trunc2, n_classes, n_classes2



def get_encoded_dataset(dataset, tokenizer, 
                                         max_length):

    def data_preprocesing(row):
        # Токенизация
        return tokenizer(row['text'], truncation=True, max_length=max_length)
    
    tokenized_dataset = dataset.map(data_preprocesing, batched=True)

    tokenized_dataset.set_format("torch")
    return tokenized_dataset

def prepair_test_dataset(df_test, level,
                           max_number_tokens=512, 
                           pre_trained_model_name='DeepPavlov/rubert-base-cased'):
    
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name, do_lower_case = True)
    
    dataset_test = Dataset.from_pandas(df_test[["text",f'target_coded{level}']].\
                                    rename(columns={f'target_coded{level}': "label"}))


    print("Подготовка тестовых данных:")

    dataset_test = get_encoded_dataset(dataset_test, tokenizer=tokenizer,
                                                                max_length=max_number_tokens)

    
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    return dataset_test, tokenizer, collate_fn

    

def prepair_datasets(df, df_test, n_classes, level,
                           max_number_tokens=512, 
                           pre_trained_model_name='DeepPavlov/rubert-base-cased'):
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name, do_lower_case = True)

    
    # Ищем элемнты, c list из target_coded, который встречается только 1 раз
    list_of_keys_less_than_two = []
    for key, val in Counter(df[f'target_coded{level}'].apply(lambda x: str(x))).items():
        if val < 2:
            list_of_keys_less_than_two.append(key)
    #Отделяем элементы датасета c list из target_coded, который встречается только 1 раз
    df_trunc_single_example = df[df[f'target_coded{level}'].apply(lambda x: str(x) 
                                                                        in list_of_keys_less_than_two)]
    df_trunc_no_less_than_two = df[df[f'target_coded{level}'].apply(lambda x: str(x) 
                                                                        not in list_of_keys_less_than_two)]
    # Создаем стратифицированную выборку для обучения и валидации
    train_df_0, valid_df= train_test_split(df_trunc_no_less_than_two, 
                                        stratify=df_trunc_no_less_than_two[f'target_coded{level}'].apply(lambda x: str(x)),
                                            test_size=0.2)
    # Добавляем в обучающую выборку элементы начального датасета c list из target_coded, который встречается только 1 раз
    train_df = pd.concat([train_df_0, df_trunc_single_example], ignore_index=True)

    number_of_rows = train_df[f'target_coded{level}'].shape[0]
    number_per_class_2 = np.array([train_df[f'target_coded{level}'].apply(lambda x: x[index]).sum() 
                                for index in range(n_classes)])

    # Cчитаем веса каждого классов
    weights_per_class = torch.tensor(number_of_rows / (number_per_class_2 * n_classes))
    print("Веса для кажого класса: ", weights_per_class)

    dataset_train = Dataset.from_pandas(train_df[["text",f'target_coded{level}']].\
                                        rename(columns={f'target_coded{level}': "label"}))
    
    dataset_valid = Dataset.from_pandas(valid_df[["text",f'target_coded{level}']].\
                                    rename(columns={f'target_coded{level}': "label"}))
    
    dataset_test = Dataset.from_pandas(df_test[["text",f'target_coded{level}']].\
                                    rename(columns={f'target_coded{level}': "label"}))
    print("Подготовка тренировочных данных:")

    dataset_train= get_encoded_dataset(dataset_train, tokenizer=tokenizer,
                                                                   max_length=max_number_tokens)
    print("Подготовка валидационных данных:")


    dataset_valid = get_encoded_dataset(dataset_valid, tokenizer=tokenizer,
                                                                max_length=max_number_tokens)
    print("Подготовка тестовых данных:")

    dataset_test = get_encoded_dataset(dataset_test, tokenizer=tokenizer,
                                                                max_length=max_number_tokens)

    
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    return dataset_train, dataset_valid, dataset_test, tokenizer, collate_fn, weights_per_class
        # loss_fuction_for_multiclass_classification 

def prepair_model(n_classes,
                  pre_trained_model_name='DeepPavlov/rubert-base-cased',
                  r=16,
                  lora_alpha=32,
                  lora_dropout=0.05):
    model = AutoModelForSequenceClassification.from_pretrained(pre_trained_model_name,
                                                               problem_type="multi_label_classification",
                                                               num_labels=n_classes)
    print(model)
    # for param in model.parameters():
    #     param.requires_grad = False
    # lora для модели
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules= ["query", "key"],
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        modules_to_save=["classifier"]
    )
    model_peft = get_peft_model(model, config)
    model_peft.print_trainable_parameters()

    return model_peft

# Функция подсчета всех метрик при валидации
def prepair_compute_metrics(n_classes):

    multilabel_auroc_micro =  MultilabelAUROC(num_labels=n_classes,
                                               average="micro", thresholds=5)
    multilabel_auroc_macro =  MultilabelAUROC(num_labels=n_classes, 
                                              average="macro", thresholds=5)
    multilabel_auroc_weighted =  MultilabelAUROC(num_labels=n_classes, 
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
        preds = torch.sigmoid(torch.tensor(pred.predictions).float())# Принимем сигмоду для получения вероятностей

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
            'accuracy_weighted_0.5':accuracy_weighted,
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


class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            class_weights = class_weights.clone().detach().type(torch.float).to(self.args.device)
            self.class_weights = class_weights
        else:
            self.class_weights = None 

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(self.args.device).float()#.long()

        outputs = model(**inputs)

        logits = outputs.get('logits')

        if self.class_weights is not None:
            loss = torch.nn.BCEWithLogitsLoss(weight=self.class_weights).to(self.args.device)
        else:
            loss =  torch.nn.BCEWithLogitsLoss().to(self.args.device)

        loss_res = loss(logits, labels)

        return (loss_res, outputs) if return_outputs else loss_res

def save_parameters(dir_name, 
                    number_of_delteted_values, minimal_number_of_elements_RGNTI2,
                    minimal_number_of_words, max_number_tokens, pre_trained_model_name,
                    r, lora_alpha, lora_dropout, epoch, batch_size,
                    weight_decay, warmup_steps,
                    fp16, optim):
    
    sett = TrainSettings()
    sett.settings["number_of_delteted_values"] = number_of_delteted_values
    sett.settings["minimal_number_of_elements_RGNTI2"] = minimal_number_of_elements_RGNTI2
    sett.settings["minimal_number_of_words"] = minimal_number_of_words
    sett.settings["max_number_tokens"] = max_number_tokens
    sett.settings["pre_trained_model_name"] = pre_trained_model_name
    sett.settings["r"] = r
    sett.settings["lora_alpha"] = lora_alpha
    sett.settings["lora_dropout"] = lora_dropout
    sett.settings["epoch"] = epoch
    sett.settings["batch_size"] = batch_size
    sett.settings["weight_decay"] = weight_decay
    sett.settings["warmup_steps"] = warmup_steps
    sett.settings["fp16"] = fp16
    sett.settings["optim"] = optim
    sett.save(path = dir_name)

def test_predictons(preds, test_dataset_labels, dir_name, 
                       n_classes, level,
                       grnti_path="", change_for_topk_y=True):
    treshold_list = [0.1 + 0.05 * x for x in range(0, 18, 1)]
    f1_score_macro_list = []
    f1_score_micro_list = []
    f1_score_weighted_list = []


    precision_macro_list = []
    precision_micro_list = []
    precision_weighted_list = []

    recall_macro_list = []
    recall_micro_list = []
    recall_weighted_list = []
    
    best_treshold = None
    best_metrics  = dict()
    best_some_f1 = 0
    preds = torch.tensor(preds)

    for treshold in tqdm(treshold_list):
        multilabel_f1_score_macro = MultilabelF1Score(num_labels=n_classes, average='macro', 
                                                      threshold=treshold)
        multilabel_precision_macro = MultilabelPrecision(num_labels=n_classes, average='macro', 
                                                      threshold=treshold)
        multilabel_recall_macro = MultilabelRecall(num_labels=n_classes, average='macro', 
                                                      threshold=treshold)
        
        f1_score_macro_list.append(multilabel_f1_score_macro(preds, 
                                                            torch.tensor(test_dataset_labels)))
        precision_macro_list.append(multilabel_precision_macro(preds, 
                                                            torch.tensor(test_dataset_labels)))
        recall_macro_list.append(multilabel_recall_macro(preds, 
                                                            torch.tensor(test_dataset_labels)))
        ####
        multilabel_f1_score_micro = MultilabelF1Score(num_labels=n_classes, average='micro', threshold=treshold)
        multilabel_precision_micro = MultilabelPrecision(num_labels=n_classes, average='micro', 
                                                      threshold=treshold)
        multilabel_recall_micro = MultilabelRecall(num_labels=n_classes, average='micro', 
                                                      threshold=treshold)
        
        f1_score_micro_list.append(multilabel_f1_score_micro(preds, #torch.tensor(test_predictions_level1_2)
                                                            torch.tensor(test_dataset_labels)))
        
        precision_micro_list.append(multilabel_precision_micro(preds, 
                                                            torch.tensor(test_dataset_labels)))
        recall_micro_list.append(multilabel_recall_micro(preds, 
                                                            torch.tensor(test_dataset_labels)))

        ####

        multilabel_f1_score_weighted = MultilabelF1Score(num_labels=n_classes, average='weighted',
                                                        threshold=treshold)
        multilabel_precision_weighted  = MultilabelPrecision(num_labels=n_classes, average='weighted', 
                                                      threshold=treshold)
        multilabel_recall_weighted = MultilabelRecall(num_labels=n_classes, average='weighted', 
                                                      threshold=treshold)
        

        f1_score_weighted_list.append(multilabel_f1_score_weighted(preds,#torch.tensor(test_predictions_level1_2), 
                                                            torch.tensor(test_dataset_labels)))
        precision_weighted_list.append(multilabel_precision_weighted(preds, 
                                                            torch.tensor(test_dataset_labels)))
        recall_weighted_list.append(multilabel_recall_weighted(preds, 
                                                            torch.tensor(test_dataset_labels)))
        ####


        
        sum_f1 = f1_score_macro_list[-1] + f1_score_micro_list[-1] + f1_score_weighted_list[-1]

        if sum_f1 > best_some_f1:
            best_some_f1 = sum_f1
            best_treshold = treshold
            best_metrics["best_treshold"] = best_treshold

            best_metrics["f1_macro"] = f1_score_macro_list[-1].detach().item()
            best_metrics["f1_micro"] = f1_score_micro_list[-1].detach().item()
            best_metrics["f1_weighted"] = f1_score_weighted_list[-1].detach().item()

            best_metrics["precision_macro"] = precision_macro_list[-1].detach().item()
            best_metrics["precision_micro"] = precision_micro_list[-1].detach().item()
            best_metrics["precision_weighted"] = precision_weighted_list[-1].detach().item()

            best_metrics["recall_macro"] = recall_macro_list[-1].detach().item()
            best_metrics["recall_micro"] = recall_micro_list[-1].detach().item()
            best_metrics["recall_weighted"] = recall_weighted_list[-1].detach().item()


    multilabel_f1_score_none = MultilabelF1Score(num_labels=n_classes, average='none',
                                                    threshold=best_treshold)
    multilabel_f1_score_none_res = multilabel_f1_score_none(preds, torch.tensor(test_dataset_labels))

    with open(grnti_path + f'my_grnti{level}_int.json', "r") as code_file:
        grnti_mapping_dict_true_numbers = json.load(code_file) # Загружаем файл с кодами 
    grnti_mapping_dict_true_numbers_reverse = {y: x for x, y in 
                                               grnti_mapping_dict_true_numbers.items()}
    

    df_rubrics = pd.DataFrame({"№":[grnti_mapping_dict_true_numbers_reverse[key]
                                    for key in range(n_classes)], 
                                    "F1":torch.round(multilabel_f1_score_none_res, decimals=2)})
    df_rubrics.sort_values(by=['№'], 
                           ascending=True).to_csv(dir_name + "threshold_№_F1_sorted_by_№.csv",
                                                   index=False) 

    df_rubrics.sort_values(by=['F1'], 
                           ascending=False).to_csv(dir_name + "threshold_№_F1_sorted_by_F1.csv", 
                                                   index=False) 
    
    preds_best_treshold = torch.sum(preds >= best_treshold, axis = 1)
    
    preds_best_treshold_no_zeros = preds_best_treshold[preds_best_treshold > 0.99]


    print("Cтатистика количества пркдсказываемых классов при заданном threshold:")
    print("Среднее число предсказываемых классов для одной статьи, для которой получено предсказание",
            torch.sum(preds_best_treshold_no_zeros)/preds_best_treshold_no_zeros.shape[0])
    print("Минимальное число предсказываемых классов для одной статьи, для которой получено предсказание",
            torch.min(preds_best_treshold_no_zeros))
    print("Максимальное число предсказываемых классов для одной статьи, для которой получено предсказание",
            torch.max(preds_best_treshold_no_zeros))
    print("Доля статей без предсказанного класса:", 
          1 - preds_best_treshold_no_zeros.shape[0]/ preds_best_treshold.shape[0])
    plt.figure()

    plt.plot(treshold_list, f1_score_macro_list, label = "macro")
    plt.plot(treshold_list, f1_score_micro_list, label = "micro")
    plt.plot(treshold_list, f1_score_weighted_list, label = "weighted")
    plt.xticks(treshold_list, rotation=70)
    plt.title("Зависимость f1_score от threshold")
    plt.xlabel("threshold")
    plt.ylabel("f1_score")
    plt.legend()
    plt.grid()
    plt.savefig(dir_name + "Зависимость f1_score от threshold.png",
                    bbox_inches='tight')
    plt.close()
    plt.figure()


    plt.figure()
    plt.plot(treshold_list, precision_macro_list, label = "macro")
    plt.plot(treshold_list, precision_micro_list, label = "micro")
    plt.plot(treshold_list, precision_weighted_list, label = "weighted")
    plt.xticks(treshold_list, rotation=70)
    plt.title("Зависимость precision от threshold")
    plt.xlabel("threshold")
    plt.ylabel("precision")
    plt.legend()
    plt.grid()
    plt.savefig(dir_name + "Зависимость precision от threshold.png",
                    bbox_inches='tight')
    plt.close()

    plt.figure()
    print("recall_micro_list threshold:", recall_micro_list)
    print("recall_macro_list threshold:", recall_macro_list)
    print("recall_wighted_list threshold:", recall_weighted_list)
    plt.plot(treshold_list, recall_macro_list, label = "macro")
    plt.plot(treshold_list, recall_micro_list, label = "micro")
    plt.plot(treshold_list, recall_weighted_list, label = "weighted")
    plt.xticks(treshold_list, rotation=70)
    plt.title("Зависимость recall от threshold")
    plt.xlabel("threshold")
    plt.ylabel("recall")
    plt.legend()
    plt.grid()
    plt.savefig(dir_name + "Зависимость recall от threshold.png",
                    bbox_inches='tight')
    plt.close()


    plt.figure()
    plt.plot(treshold_list, [torch.sum(torch.sum(preds >= treshold, axis = 1) < 0.1) / preds.shape[0] for 
                             treshold in tqdm(treshold_list)])
    plt.xticks(treshold_list, rotation=70)
    plt.title("Зависимость f1_score от threshold")
    plt.xlabel("threshold")
    plt.ylabel("Доля элементов с непредсказанным классом ")
    plt.legend()
    plt.grid()
    plt.savefig(dir_name + "Зависимость доли элементов с непредсказанным классом от threshold.png",
                    bbox_inches='tight')
    plt.close()
    with open(dir_name + "best_metrics_threshold.json", "w") as outfile:
        json.dump(best_metrics, outfile)

    ###Часть с 1-м, 2-м, 3-м
    top_k_list = list(range(1, 4))
    f1_score_macro_list = []
    f1_score_micro_list = []
    f1_score_weighted_list = []

    precision_macro_list = []
    precision_micro_list = []
    precision_weighted_list = []

    recall_macro_list = []
    recall_micro_list = []
    recall_weighted_list = []


    best_top_k = None
    best_metrics  = dict()
    best_some_f1 = 0
    test_dataset_labels = torch.tensor(test_dataset_labels, dtype=float)

    for top_k in tqdm(top_k_list):
        pred_for_top_k = torch.zeros(preds.shape, dtype=float)  
        labels_for_top_k = torch.zeros(preds.shape, dtype=float)  

        top_indeces = torch.topk(preds, top_k).indices

        preds_range = torch.arange(pred_for_top_k.size(0)).unsqueeze(1)

        pred_for_top_k[preds_range, top_indeces] = 1.
        if change_for_topk_y:

            labels_for_top_k[preds_range, top_indeces] = test_dataset_labels[preds_range, top_indeces]
        else:
            labels_for_top_k = test_dataset_labels
        ###
        multilabel_f1_score_macro = MultilabelF1Score(num_labels=n_classes, average='macro')

        multilabel_precision_macro = MultilabelPrecision(num_labels=n_classes, average='macro')
        multilabel_recall_macro = MultilabelRecall(num_labels=n_classes, average='macro')
        
        f1_score_macro_list.append(multilabel_f1_score_macro(pred_for_top_k, #torch.tensor(test_predictions_level1_2)
                                                            labels_for_top_k))
        precision_macro_list.append(multilabel_precision_macro(pred_for_top_k, #torch.tensor(test_predictions_level1_2)
                                                            labels_for_top_k))
        recall_macro_list.append(multilabel_recall_macro(pred_for_top_k, #torch.tensor(test_predictions_level1_2)
                                                            labels_for_top_k))

        ##
        multilabel_f1_score_micro = MultilabelF1Score(num_labels=n_classes, average='micro')

        multilabel_precision_micro = MultilabelPrecision(num_labels=n_classes, average='micro')
        multilabel_recall_micro = MultilabelRecall(num_labels=n_classes, average='micro')
        

        f1_score_micro_list.append(multilabel_f1_score_micro(pred_for_top_k, #torch.tensor(test_predictions_level1_2)
                                                            labels_for_top_k))
        
        precision_micro_list.append(multilabel_precision_micro(pred_for_top_k, #torch.tensor(test_predictions_level1_2)
                                                            labels_for_top_k))
        recall_micro_list.append(multilabel_recall_micro(pred_for_top_k, #torch.tensor(test_predictions_level1_2)
                                                            labels_for_top_k))

        ##
        multilabel_f1_score_weighted = MultilabelF1Score(num_labels=n_classes, average='weighted')
        multilabel_precision_weighted  = MultilabelPrecision(num_labels=n_classes, average='weighted')
        multilabel_recall_weighted = MultilabelRecall(num_labels=n_classes, average='weighted')
        

        
        f1_score_weighted_list.append(multilabel_f1_score_weighted(pred_for_top_k,#torch.tensor(test_predictions_level1_2), 
                                                            labels_for_top_k))
        precision_weighted_list.append(multilabel_precision_weighted(pred_for_top_k,#torch.tensor(test_predictions_level1_2), 
                                                            labels_for_top_k))
        recall_weighted_list.append(multilabel_recall_weighted(pred_for_top_k,#torch.tensor(test_predictions_level1_2), 
                                                            labels_for_top_k))

        ##
        
        sum_f1 = f1_score_macro_list[-1] + f1_score_micro_list[-1] + f1_score_weighted_list[-1]
        
        sum_f1 = f1_score_macro_list[-1] + f1_score_micro_list[-1] + f1_score_weighted_list[-1]

        if sum_f1 > best_some_f1:
            best_some_f1 = sum_f1
            best_top_k = top_k
            best_metrics["best_top_k"] = best_top_k

            best_metrics["f1_macro"] = f1_score_macro_list[-1].detach().item()
            best_metrics["f1_micro"] = f1_score_micro_list[-1].detach().item()
            best_metrics["f1_weighted"] = f1_score_weighted_list[-1].detach().item()

            best_metrics["precision_macro"] = precision_macro_list[-1].detach().item()
            best_metrics["precision_micro"] = precision_micro_list[-1].detach().item()
            best_metrics["precision_weighted"] = precision_weighted_list[-1].detach().item()

            best_metrics["recall_macro"] = recall_macro_list[-1].detach().item()
            best_metrics["recall_micro"] = recall_micro_list[-1].detach().item()
            best_metrics["recall_weighted"] = recall_weighted_list[-1].detach().item()

    pred_for_top_k = torch.zeros(preds.shape, dtype=float)  
    labels_for_top_k = torch.zeros(preds.shape,  dtype=float)  

    top_indeces = torch.topk(preds, best_top_k).indices
    preds_range = torch.arange(pred_for_top_k.size(0)).unsqueeze(1)


    pred_for_top_k[preds_range, top_indeces] = 1.
    if change_for_topk_y:
        labels_for_top_k[preds_range, top_indeces] = test_dataset_labels[preds_range, top_indeces]
    else:
        labels_for_top_k = test_dataset_labels
    

    multilabel_f1_score_none = MultilabelF1Score(num_labels=n_classes, average='none')
    multilabel_f1_score_none_res = multilabel_f1_score_none(pred_for_top_k, labels_for_top_k)

    with open(grnti_path + f'my_grnti{level}_int.json', "r") as code_file:
        grnti_mapping_dict_true_numbers = json.load(code_file) # Загружаем файл с кодами 
    grnti_mapping_dict_true_numbers_reverse = {y: x for x, y in 
                                               grnti_mapping_dict_true_numbers.items()}
    

    df_rubrics = pd.DataFrame({"№":[grnti_mapping_dict_true_numbers_reverse[key]
                                    for key in range(n_classes)], 
                                    "F1":torch.round(multilabel_f1_score_none_res, decimals=2)})
    df_rubrics.sort_values(by=['№'], 
                           ascending=True).to_csv(dir_name + "top_k_№_F1_sorted_by_№.csv",
                                                   index=False) 

    df_rubrics.sort_values(by=['F1'], 
                           ascending=False).to_csv(dir_name + "top_k_№_F1_sorted_by_F1.csv", 
                                                   index=False) 
    
    preds_best_treshold = torch.sum(preds > 0., axis = 1)
    
    preds_best_treshold_no_zeros = preds_best_treshold[preds_best_treshold > 0.99]


    print("Cтатистика количества пркдсказываемых классов при не заданном threshold:")
    print("Среднее число предсказываемых классов для одной статьи, для которой получено предсказание",
            torch.sum(preds_best_treshold_no_zeros)/preds_best_treshold_no_zeros.shape[0])
    print("Минимальное число предсказываемых классов для одной статьи, для которой получено предсказание",
            torch.min(preds_best_treshold_no_zeros))
    print("Максимальное число предсказываемых классов для одной статьи, для которой получено предсказание",
            torch.max(preds_best_treshold_no_zeros))
    print("Доля статей без предсказанного класса:", 
          1 - preds_best_treshold_no_zeros.shape[0]/ preds_best_treshold.shape[0])
    
    plt.figure()
    plt.plot(top_k_list, f1_score_macro_list, label = "macro")
    plt.plot(top_k_list, f1_score_micro_list, label = "micro")
    plt.plot(top_k_list, f1_score_weighted_list, label = "weighted")

    print("f1_top_k_macro", np.round(f1_score_macro_list, decimals=2))
    print("f1_top_k_maicro", np.round(f1_score_micro_list, decimals=2))
    print("f1_top_k_weighted", np.round(f1_score_weighted_list, decimals=2))

    plt.xticks(top_k_list)
    plt.title("Зависимость f1_score от числа наиболее вероятных классов")
    plt.xlabel("Число наиболее вероятных классов")
    plt.ylabel("f1_score")
    plt.legend()
    plt.grid()
    plt.savefig(dir_name + "Зависимость f1_score от числа наиболее вероятных классов.png",
                    bbox_inches='tight')
    plt.close()


    plt.figure()
    plt.plot(top_k_list, precision_macro_list, label = "macro")
    plt.plot(top_k_list, precision_micro_list, label = "micro")
    plt.plot(top_k_list, precision_weighted_list, label = "weighted")


    print("precision_top_k_macro", np.round(precision_macro_list, decimals=2))
    print("precision_top_k_maicro", np.round(precision_micro_list, decimals=2))
    print("precision_top_k_weighted", np.round(precision_weighted_list, decimals=2))
    
    plt.xticks(top_k_list)
    plt.title("Зависимость precision от числа наиболее вероятных классов")
    plt.xlabel("Число наиболее вероятных классов")
    plt.ylabel("precision")
    plt.legend()
    plt.grid()
    plt.savefig(dir_name + "Зависимость precision от числа наиболее вероятных классов.png",
                    bbox_inches='tight')
    plt.close()


    print("recall_top_k_macro:", torch.stack(recall_macro_list) )
    print("recall_top_k_micro:", torch.stack(recall_micro_list))
    print("recall_top_k_weighted:", torch.stack(recall_weighted_list))

    plt.figure()
    plt.plot(top_k_list, torch.stack(recall_macro_list).to(int), label = "macro")
    plt.plot(top_k_list, torch.stack(recall_micro_list).to(int), label = "micro")
    plt.plot(top_k_list, torch.stack(recall_weighted_list).to(int), label = "weighted")
    plt.xticks(top_k_list)
    plt.title("Зависимость recall от числа наиболее вероятных классов")
    plt.xlabel("Число наиболее вероятных классов")
    plt.ylabel("recall")
    plt.legend()
    plt.grid()
    plt.savefig(dir_name + "Зависимость recall от числа наиболее вероятных классов.png",
                    bbox_inches='tight')
    # plt.close()

    with open(dir_name + "best_metrics_top_k.json", "w") as outfile:
        json.dump(best_metrics, outfile)

    def eval_step(engine, batch):
        return batch

    default_evaluator = Engine(eval_step)
    metric = ClassificationReport(output_dict=True, is_multilabel=True)
    weighted_metric_precision = Precision(average='weighted', is_multilabel=True)
    weighted_metric_recall= Recall(average='weighted', is_multilabel=True)

    preds = (preds >= 0.5).int() # torch.tensor(
    input = torch.tensor(test_dataset_labels).int()

    precision = Precision(average=False, is_multilabel=True)
    recall = Recall(average=False, is_multilabel=True)
    F1 = precision * recall * 2 / (precision + recall + 1e-20)
    freq = torch.tensor([sum(input[:, i]) for i in range(input.shape[1])])#.tolist()
    # weights_per_class = input.shape[0] / (torch.tensor([el if el > 0 else 1 for el in freq])* input.shape[1])
    F1_wieghted = MetricsLambda(lambda t: torch.sum(t * freq).item() / input.shape[0], F1) # 

    metric.attach(default_evaluator, "cr")
    weighted_metric_precision.attach(default_evaluator, "weighted precision")
    weighted_metric_recall.attach(default_evaluator, "weighted recall")
    F1_wieghted.attach(default_evaluator, "weighted F1")



    state = default_evaluator.run([[preds, input]])
    result = state.metrics['cr']
    result['weighted precision'] = state.metrics['weighted precision']

    result['weighted recall'] = state.metrics['weighted recall']
    result['weighted F1'] = state.metrics['weighted F1']

    with open(dir_name + "test_results.json", "w") as outfile:
        json.dump(result, outfile)