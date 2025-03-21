import os
import json
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):

    def __init__(self, raw_data, label_dict, tokenizer, model_name, method):
        self.label_list = list(label_dict.keys()) if method not in ['ce', 'scl'] else []
        self.sep_token = ['[SEP]'] if model_name == 'bert' or model_name == 'rubert' else ['</s>']
        self.dataset = list()
        
        for data in raw_data:
            tokens = data['text'].lower().split(' ')
            # Преобразуем метки в бинарный вектор
            
            label_vector = [0] * len(label_dict)
            # print(len(data['label']))
            for label in data['label']:#labels  # Предполагаем, что data['labels'] это список меток
                label_vector[label_dict[label]] = 1

            
            self.dataset.append((self.label_list + self.sep_token + tokens, label_vector))#label_vector

    def __getitem__(self, index):
        tokens, label_vector = self.dataset[index]
        return tokens, label_vector

    def __len__(self):
        return len(self.dataset)


def my_collate(batch, tokenizer, method, num_classes):
    tokens, label_ids = map(list, zip(*batch))
    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=256,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    # if method not in ['ce', 'scl']:
    #     positions = torch.zeros_like(text_ids['input_ids'])
    #     positions[:, num_classes:] = torch.arange(0, text_ids['input_ids'].size(1)-num_classes)
    #     text_ids['position_ids'] = positions
    # return text_ids, torch.tensor(label_ids)

    # Добавляем position_ids, если метод не 'ce' или 'scl'
    if method not in ['ce', 'scl']:
        positions = torch.zeros_like(text_ids['input_ids'])
        positions[:, num_classes:] = torch.arange(0, text_ids['input_ids'].size(1) - num_classes)
        text_ids['position_ids'] = positions
    
    # Преобразуем label_ids в тензор
    label_ids = torch.tensor(label_ids, dtype=torch.float32)  # Используем float32 для многозадачной классификации
    
    return text_ids, label_ids


def load_data(dataset, data_dir, tokenizer, train_batch_size, test_batch_size, model_name, method, workers):
    if dataset == 'sst2':
        train_data = json.load(open(os.path.join(data_dir, 'SST2_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SST2_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'trec':
        train_data = json.load(open(os.path.join(data_dir, 'TREC_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'TREC_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'description': 0, 'entity': 1, 'abbreviation': 2, 'human': 3, 'location': 4, 'numeric': 5}
    elif dataset == 'cr':
        train_data = json.load(open(os.path.join(data_dir, 'CR_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'CR_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'subj':
        train_data = json.load(open(os.path.join(data_dir, 'SUBJ_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SUBJ_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'subjective': 0, 'objective': 1}
    elif dataset == 'pc':
        train_data = json.load(open(os.path.join(data_dir, 'procon_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'procon_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == "rucola":
        train_data = json.load(open(os.path.join(data_dir, 'train_rucola.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'validation_rucola.json'), 'r', encoding='utf-8'))
        label_dict = {'No error':0,
                      'Morphology':1,
                      'Semantics':2,
                      'Syntax':3,
                      'Hallucination':4}
        
    elif dataset == "grnit_level_1":
        train_data = json.load(open(os.path.join(data_dir, 'train_grnti.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'test_grnti.json'), 'r', encoding='utf-8'))
        label_dict ={'Астрономия': 0,
                    'Механика': 1,
                    'Сельское и лесное хозяйство': 2,
                    'Транспорт': 3,
                    'Организация и управление': 4,
                    'Биология': 5,
                    'Информатика': 6,
                    'Математика': 7,
                    'Метрология': 8,
                    'Космические исследования': 9,
                    'Автоматика. Вычислительная техника': 10,
                    'Связь': 11,
                    'Охрана окружающей среды. Экология человека': 12,
                    'Машиностроение': 13,
                    'Медицина и здравоохранение': 14,
                    'Химическая технология. Химическая промышленность': 15,
                    'Электротехника': 16,
                    'Общие и комплексные проблемы технических и прикладных наук и отрас-лей народного хозяйства': 17,
                    'Геофизика': 18,
                    'Химия': 19,
                    'Электроника. Радиотехника': 20,
                    'Физика': 21,
                    'Энергетика': 22,
                    'Металлургия': 23,
                    'Горное дело': 24,
                    'Геология': 25,
                    'Водное хозяйство': 26,
                    'География': 27,
                    'Кибернетика': 28,
                    'Пищевая промышленность': 29,
                    'Экономика и экономические науки': 30,
                    'Биотехнология': 31}

    else:
        raise ValueError('unknown dataset')
    trainset = MyDataset(train_data, label_dict, tokenizer, model_name, method)
    # print("train_data complieted")

    testset = MyDataset(test_data, label_dict, tokenizer, model_name, method)
    collate_fn = partial(my_collate, tokenizer=tokenizer, method=method, num_classes=len(label_dict))
    train_dataloader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn, pin_memory=True)
    test_dataloader = DataLoader(testset, test_batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn, pin_memory=True)
    return train_dataloader, test_dataloader
