import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm2
import json
import torch
from peft import TaskType, LoraConfig, get_peft_model
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from collections import Counter
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MultilabelF1Score, MultilabelAccuracy, MultilabelAUROC
from datasets import Dataset

tqdm2.pandas()

def get_grnti1_2_BERT_dataframes(file_path, number_of_delteted_values, 
                                 minimal_number_of_elements_RGNTI2,
                                 minimal_number_of_words,
                                 dir_name=None):
    df = pd.read_csv(file_path + "\\train_ru.csv", sep='\t', encoding='cp1251')
    df = df.loc[df['RGNTI'].apply(lambda x: re.findall("\d+",x)!=[])] # Пропускаем строки без класса
    df_test = pd.read_csv(file_path + "\\test_ru.csv", sep='\t', encoding='cp1251',
                        error_bad_lines=False)
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

    pd.value_counts(np.concatenate(df['target'].values))[-number_of_delteted_values:].plot.bar()
    plt.xlabel("RGNTI 1")
    plt.ylabel("Количество элементов")
    plt.title("Количество удаляемых текстов из датасета для 1-ого уровня ГРНТИ")
    # plt.show()
    if dir_name:
        plt.savefig(dir_name + "Количество удаляемых текстов из датасета для 1-ого уровня ГРНТИ.png",
                    bbox_inches='tight')

    pd.value_counts(np.concatenate(df['target'].values))[:-number_of_delteted_values].plot.bar()
    plt.xlabel("RGNTI 1")
    plt.ylabel("Количество элементов")
    plt.title("Количество элементов, остающихся в датасете")
    # plt.show()
    if dir_name:
        plt.savefig(dir_name + "Количество элементов, остающихся в датасете.png",
                    bbox_inches='tight')


    list_of_few_values = pd.value_counts(np.concatenate(
        df['target'].values))[:-number_of_delteted_values].index.to_list() 
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

    for target_2_val in tqdm(unique_vals):
        needed_taget2 = df_trunc['target_2'].apply(lambda x: [re.findall(f"{target_2_val}.\d+",el)[0] for el 
                                                    in x if re.findall(f"{target_2_val}.\d+",el)])
        concatenated_list_target2 = pd.value_counts(np.concatenate(np.array([el for el
                                                    in needed_taget2.values.tolist() if el], dtype="object")))
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

    with open("my_grnti1_int.json", "w") as outfile:
        json.dump(dict_Vinit_code_int, outfile)


    with open('my_grnti1_int.json', "r") as code_file:
        grnti_mapping_dict = json.load(code_file) # Загружаем файл с кодами 
    n_classes = len(grnti_mapping_dict)


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

    return df_trunc2, df_test_trunc2, n_classes

def get_input_ids_attention_masks_token_type_labels(df, tokenizer, max_len):
    # Токенизация 
    input_ids = []
    attention_masks = []
    token_type_ids =[]
    # Для каждого тектса...
    for sent in tqdm(df['text']):
        encoded_dict = tokenizer.encode_plus(
                sent,
                max_length = max_len,
                return_tensors = 'pt',
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask = True,
                padding = 'max_length')
        # Добавляем закодированный текст в list.
        input_ids.append(encoded_dict['input_ids'])
        # Добавляем attention mask (Отделяем padding от non-padding токенов).
        attention_masks.append(encoded_dict['attention_mask'])
        #Добавляем token_type_ids, тк у нас есть [SEP] в тексах 
        token_type_ids.append(encoded_dict['token_type_ids'])

    # Переводим листы в тензоры.
    input_ids= torch.cat(input_ids, dim=0)
    attention_masks= torch.cat(attention_masks, dim=0)
    token_type_ids= torch.cat(token_type_ids, dim=0)
    labels= torch.tensor(df['target_coded'].to_list()).float()

    return input_ids, attention_masks, token_type_ids, labels


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

def prepair_datasets(df, df_test, n_classes,
                           max_number_tokens=512, 
                           pre_trained_model_name='DeepPavlov/rubert-base-cased'):
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name, do_lower_case = True)

    
    # Ищем элемнты, c list из target_coded, который встречается только 1 раз
    list_of_keys_less_than_two = []
    for key, val in Counter(df['target_coded'].apply(lambda x: str(x))).items():
        if val < 2:
            list_of_keys_less_than_two.append(key)
    #Отделяем элементы датасета c list из target_coded, который встречается только 1 раз
    df_trunc_single_example = df[df['target_coded'].apply(lambda x: str(x) 
                                                                        in list_of_keys_less_than_two)]
    df_trunc_no_less_than_two = df[df['target_coded'].apply(lambda x: str(x) 
                                                                        not in list_of_keys_less_than_two)]
    # Создаем стратифицированную выборку для обучения и валидации
    train_df_0, valid_df= train_test_split(df_trunc_no_less_than_two, 
                                        stratify=df_trunc_no_less_than_two['target_coded'].apply(lambda x: str(x)),
                                            test_size=0.2)
    # Добавляем в обучающую выборку элементы начального датасета c list из target_coded, который встречается только 1 раз
    train_df = pd.concat([train_df_0, df_trunc_single_example], ignore_index=True)

    number_of_rows = train_df['target_coded'].shape[0]
    number_per_class_1 = np.array([train_df['target_coded'].apply(lambda x: x[index]).sum() 
                                for index in range(n_classes)])

    # Cчитаем веса каждого классов
    weights_per_class = number_of_rows / (number_per_class_1 * n_classes)
    print("Веса для кажого класса: ", weights_per_class)
    print("Подготовка тренировочных данных:")
    input_ids_train, attention_masks_train,\
    token_type_ids_train,\
    labels_train = get_input_ids_attention_masks_token_type_labels(train_df, tokenizer=tokenizer,
                                                                   max_len=max_number_tokens)
    print("Подготовка валидационных данных:")
    input_ids_validation, attention_masks_validation,\
    token_type_ids_validation,\
    labels_validation = get_input_ids_attention_masks_token_type_labels(valid_df, tokenizer=tokenizer,
                                                                   max_len=max_number_tokens)
    print("Подготовка тестовых данных:")
    input_ids_test, attention_masks_test,\
    token_type_ids_test,\
    labels_test = get_input_ids_attention_masks_token_type_labels(df_test, tokenizer=tokenizer,
                                                                   max_len=max_number_tokens)
    #Собираем датасеты и делаем shuffle для каждого
    SEED = 1234
    dataset_train_v2 = Dataset.from_dict({"input_ids":input_ids_train,  
                                        "attention_mask":attention_masks_train,  
                                        "labels":labels_train,
                                        "token_type_ids":token_type_ids_train}).shuffle(SEED)

    dataset_valid_v2 = Dataset.from_dict({"input_ids":input_ids_validation,  
                                        "attention_mask":attention_masks_validation,  
                                        "labels":labels_validation,
                                        "token_type_ids":token_type_ids_validation}).shuffle(SEED)
    dataset_test_v2 = Dataset.from_dict({"input_ids":input_ids_test,  
                                      "attention_mask":attention_masks_test,  
                                      "labels":labels_test,
                                      "token_type_ids":token_type_ids_test}).shuffle(SEED)
    


    #Функция потерь с учетом весов для multilabel классификации
    loss_fuction_for_multiclass_classification =\
        torch.nn.BCEWithLogitsLoss(weight = torch.tensor(weights_per_class).float()).to("cuda")
    #Чтобы использовать собственную функцию потерь создаем класс CustomTrainer

    return dataset_train_v2, dataset_valid_v2, dataset_test_v2,\
        loss_fuction_for_multiclass_classification 

def prepair_model(n_classes,
                  pre_trained_model_name='DeepPavlov/rubert-base-cased',
                  r=16,
                  lora_alpha=32,
                  lora_dropout=0.05):
    model = AutoModelForSequenceClassification.from_pretrained(pre_trained_model_name,
                                                               problem_type="multi_label_classification",
                                                               num_labels=n_classes)
    # lora для модели
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    model_peft = get_peft_model(model, config)

    return model_peft

