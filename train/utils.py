import torch
from torch import nn
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import re
from peft import LoraConfig, get_peft_model 
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


class BertDatasetTrainGrnti1(torch.utils.data.Dataset):
    def __init__(self, df, max_len, tokenizer):

        self.X= df['text'].to_list()
        self.y = df['target_coded'].to_list()
        self.tokenizer = tokenizer
        self.max_length = max_len
               
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        text = self.X[idx]
        target = self.y[idx]

        tokenizer_output = self.tokenizer.encode_plus(
            text,
            max_length = self.max_length,
            return_tensors = 'pt',
            padding = 'max_length')
        return {
            "input_ids": tokenizer_output['input_ids'].squeeze(0)[:self.max_length], 
            "mask": tokenizer_output['attention_mask'].squeeze(0)[:self.max_length],
            'targets': torch.tensor(target, dtype=torch.long)
        }

    
def clean(text): 
    text = re.sub(r"[^а-яёА-ЯЁa-zA-Z]", " ", text)
    
    Special = '@#!?+&*[]-%:/()$=><|{}^' 
    for s in Special:
        text = text.replace(s, "")
        
    return text

def truncating_geting_text_df(df, df_test, number_of_indexes_for_truncation):
    list_of_few_values = pd.value_counts(df['RGNTI1'])[-number_of_indexes_for_truncation:].index.to_list()
    df_trunc = df.query('RGNTI1 not in @list_of_few_values')
    df_test_trunc = df_test.query('RGNTI1 not in @list_of_few_values')

    unique_vals = df_trunc['RGNTI1'].unique()
    unique_vals_test = df_test_trunc['RGNTI1'].unique()

    set_targets = set(unique_vals) 
    coding =  range(len(set_targets))
    dict_Vinit_code_int = dict(zip(set_targets, coding))

    with open("source/my_grnti1_int.json", "w") as outfile: 
        json.dump(dict_Vinit_code_int, outfile)
    grnti_mapping_dict = json.load(open('source\\my_grnti1_int.json')) # некоторых значений в словаре нет

    n_classes = len(grnti_mapping_dict)

    df_trunc['target_coded'] = df_trunc['target'].apply(lambda x: grnti_mapping_dict[x])
    df_test_trunc['target_coded'] = df_test_trunc['target'].apply(lambda x: grnti_mapping_dict[x])

    df_trunc['text'] = (df_trunc['title'].apply(lambda x:x+' ' if x[-1] in 
                                                ('.', '!', '?') else x+'. ') + df_trunc['ref_txt'])
    df_test_trunc['text'] = (df_test_trunc['title'].apply(lambda x:x+' ' if x[-1] 
                                            in ('.', '!', '?') else x+'. ') + df_test_trunc['ref_txt'])


    df_trunc['text'] = df_trunc['text'].apply(lambda s : clean(s))
    df_test_trunc['text'] = df_test_trunc['text'].apply(lambda s : clean(s))

    return df_trunc, df_test_trunc, n_classes

def creating_bert_peft_lora_model_lerning(pre_trained_model_name, n_classes, r, 
                                          lora_alpha,
                                          lora_dropout,
                                          bias,
                                          task_type):

    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(pre_trained_model_name, 
                                                               num_labels=n_classes)

    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    model.classifier = nn.Linear(model.config.hidden_size, n_classes)
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type
    )

    model_peft = get_peft_model(model, config)
    return model_peft, tokenizer


def teach_model(model, dataloader, device, save_path):
    model.to(device)
    model.train()

    total_step = len(dataloader)
    correct = 0
    total = 0
    running_loss = 0.0
    loss_per_iter = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for batch_id, batch in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        inputs = batch['input_ids'].to(device)
        mask = batch['mask'].to(device)
        y_train = batch['targets'].to(device)
        y_pred = model(input_ids = inputs, attention_mask = mask)
        loss = nn.functional.cross_entropy(y_pred.logits, y_train)
        loss.backward()
        optimizer.step()
        

        # print statistics
        running_loss += loss.item()
        _,pred = torch.max(y_pred.logits, dim=1)
        correct += torch.sum(pred==y_train).item()
        total += y_train.size(0)
        loss_per_iter.append(loss.item())

        if (batch_id % 100) == 0:
            print(f'Step [{batch_id}/{total_step}], Loss: {loss_per_iter[-1]}')

    train_acc = (100 * correct / total)
    train_loss = (running_loss / total_step)    
    torch.save(model.state_dict(), save_path)
    return train_acc, train_loss, loss_per_iter