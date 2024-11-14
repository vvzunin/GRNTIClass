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
from peft import PeftConfig, PeftModel
import csv
from torch.utils.data import DataLoader

def prepair_model(n_classes, lora_model_path,
                  pre_trained_model_name='DeepPavlov/rubert-base-cased',
                  ):
    print("Подготовка модели")
    model = AutoModelForSequenceClassification.from_pretrained(pre_trained_model_name,
                                                               problem_type="multi_label_classification",
                                                               num_labels=n_classes)
    for param in model.parameters():
        param.requires_grad = False
    PeftConfig.from_pretrained(lora_model_path)
    model = PeftModel.from_pretrained(model, lora_model_path, 
                                    torch_dtype=torch.float16)

    return model


def prepair_data_level1(file_path):
    print("Подготовка данных 1 уровень")

    df_test = pd.read_csv(file_path, sep='\t', encoding='cp1251', on_bad_lines='skip')
  
    df_test['text'] = (df_test['title'].apply(lambda x:x+' [SEP] ')
                                                + df_test['ref_txt'])

    df_test['text'] = df_test['text'].apply(lambda x:str(x)+
                                                          ' [SEP] ') + df_test['kw_list']
    
    df_test = df_test.dropna(subset=['text'], axis=0)


    return df_test#, n_classes1

def prepair_data_level2(df_test, preds):

    print("Подготовка данных 2 уровень")

    with open('src\\my_grnti1_int.json', "r") as code_file:
        grnti_mapping_dict_true_numbers = json.load(code_file) # Загружаем файл с кодами 

    with open('src\\GRNTI_1_ru.json', "r", encoding='utf-8') as code_file:
        grnti_mapping_dict_true_names = json.load(code_file) # Загружаем файл с кодами 


    list_GRNTI =[]
    for el in tqdm(preds):
        list_elments = []

        for index, propab in enumerate(el):
            if propab==1:
                list_elments.append(index) 
        list_GRNTI.append(list_elments)

    print("Доля непредсказанных классов GRNTI 1 для статей:", 
      sum([not el for el in list_GRNTI])/len(list_GRNTI))
    

    grnti_mapping_dict_true_numbers_reverse = {y: x for x, y in 
                                               grnti_mapping_dict_true_numbers.items()}
    list_true_numbers_GRNTI = []
    for list_el in list_GRNTI:
        list_numbers = []
        for el in list_el:
            list_numbers.append(grnti_mapping_dict_true_numbers_reverse[el])
        list_true_numbers_GRNTI.append(list_numbers)

    list_thems = []
    for list_true in list_true_numbers_GRNTI:
        sring_per_element = ""
        for el in list_true:
            sring_per_element += grnti_mapping_dict_true_names[el] + "; "
        list_thems.append(sring_per_element)

    # df_test = df_test.iloc[:24]

    df_test['text'] = list_thems + df_test['text']#text_with_GRNTI1_names
    return df_test

def get_input_ids_attention_masks_token_type(df, tokenizer, max_len):
    # Токенизация 
    input_ids = []
    attention_masks = []
    token_type_ids =[]
    # Для каждого тектса...
    for sent in tqdm(df['text']):#text_with_GRNTI1_names
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

    return input_ids, attention_masks, token_type_ids

def collate_fn(batch):
    result = {}
    for el in batch:
        for key in el.keys():
            result.setdefault(key, []).append(el[key])
    for key in result.keys():
        result[key] = torch.tensor(result[key])
    return result
def prepair_dataset(df_test,
                           max_number_tokens=512, 
                           pre_trained_model_name='DeepPavlov/rubert-base-cased'):
    
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name, do_lower_case = True) 


    print("Подготовка тестового датасета:")
    input_ids_test, attention_masks_test,\
    token_type_ids_test = get_input_ids_attention_masks_token_type(df_test, tokenizer=tokenizer,
                                                                   max_len=max_number_tokens)

    dataset_test = Dataset.from_dict({"input_ids":input_ids_test,  
                                      "attention_mask":attention_masks_test,  
                                      "token_type_ids":token_type_ids_test})
    

    test_dataloader = DataLoader(dataset_test, batch_size=8, collate_fn=collate_fn)
    return test_dataloader


def make_predictions(model, dataset_test, device, threshold):
    print("Предсказние модели")
    model.eval()
    y_pred_list = []
    model.to(device)

    # count = 0

    for batch in tqdm(dataset_test):
            # if count == 3:
            #     break

            inputs = batch['input_ids'].to(device = device, dtype=torch.long)
            mask = batch['attention_mask'].to(device = device)

            with torch.no_grad():
                output = model(input_ids = inputs, attention_mask = mask)
            
            # Move logits and labels to CPU
            logits = output.logits.detach().cpu()

            logits_flatten = (torch.sigmoid(logits).numpy() >= threshold).tolist()

            y_pred_list.extend(logits_flatten)

            # count +=1

    return y_pred_list


def save_rubrics_names(preds, path_to_csv):

    with open('src\\my_grnti2_int.json', "r") as code_file:
            grnti_mapping_dict_true_numbers = json.load(code_file) # Загружаем файл с кодами 

    with open('src\\GRNTI_2_ru.json', "r", encoding='utf-8') as code_file:
        grnti_mapping_dict_true_names = json.load(code_file) # Загружаем файл с кодами 


    list_GRNTI =[]
    for el in tqdm(preds):
        list_elments = []

        for index, propab in enumerate(el):
            if propab==1:
                list_elments.append(index) 
        list_GRNTI.append(list_elments)

    print("Доля непредсказанных классов GRNTI 2 для статей:", 
      sum([not el for el in list_GRNTI])/len(list_GRNTI))
    

    grnti_mapping_dict_true_numbers_reverse = {y: x for x, y in 
                                               grnti_mapping_dict_true_numbers.items()}
    list_true_numbers_GRNTI = []
    for list_el in list_GRNTI:
        list_numbers = []
        for el in list_el:
            list_numbers.append(grnti_mapping_dict_true_numbers_reverse[el])
        list_true_numbers_GRNTI.append(list_numbers)

    list_thems = []
    for list_true in list_true_numbers_GRNTI:
        sring_per_element = ""
        for el in list_true:
            sring_per_element += grnti_mapping_dict_true_names[el] + "; "
        list_thems.append(sring_per_element)

    np.savetxt(path_to_csv,
            list_thems,
            delimiter =", ",
            fmt ='% s')