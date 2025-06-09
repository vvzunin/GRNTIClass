import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader


def prepair_model(
        n_classes, lora_model_path,
        pre_trained_model_name="DeepPavlov/rubert-base-cased"):

    model = BertForSequenceClassification.from_pretrained(
        pre_trained_model_name,
        problem_type="multi_label_classification",
        num_labels=n_classes,
        # use_safetensors=True
    )

    for param in model.parameters():
        param.requires_grad = False

    PeftConfig.from_pretrained(lora_model_path)
    model = PeftModel.from_pretrained(
        model, lora_model_path, torch_dtype=torch.float16)
    return model


def get_input_ids_attention_masks_token_type(df, tokenizer, max_len):
    # Токенизация
    input_ids = []
    attention_masks = []
    token_type_ids = []
    # Для каждого текста...
    for sent in df["text"]:  # text_with_GRNTI1_names
        encoded_dict = tokenizer.encode_plus(
            sent,
            max_length=max_len,
            return_tensors="pt",
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            padding="max_length",
        )
        # Добавляем закодированный текст в list.
        input_ids.append(encoded_dict["input_ids"])
        # Добавляем attention mask (Отделяем padding от non-padding токенов).
        attention_masks.append(encoded_dict["attention_mask"])
        # Добавляем token_type_ids, тк у нас есть [SEP] в тексах
        token_type_ids.append(encoded_dict["token_type_ids"])

    # Переводим листы в тензоры.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)

    return input_ids, attention_masks, token_type_ids


def collate_fn(batch):
    result = {}
    for el in batch:
        for key in el.keys():
            result.setdefault(key, []).append(el[key])
    for key in result.keys():
        result[key] = torch.tensor(result[key])
    return result


def prepair_dataset(
    df_test,
    max_number_tokens=512,
    pre_trained_model_name="DeepPavlov/rubert-base-cased",
):

    tokenizer = BertTokenizer.from_pretrained(
        pre_trained_model_name, do_lower_case=True
    )

    input_ids_test, attention_masks_test, token_type_ids_test = (
        get_input_ids_attention_masks_token_type(
            df_test, tokenizer=tokenizer, max_len=max_number_tokens
        )
    )

    dataset_test = Dataset.from_dict(
        {
            "input_ids": input_ids_test,
            "attention_mask": attention_masks_test,
            "token_type_ids": token_type_ids_test,
        }
    )

    test_dataloader = DataLoader(
        dataset_test, batch_size=8, collate_fn=collate_fn)
    return test_dataloader


def make_predictions(model, dataset_test, device):
    model.eval()
    y_pred_list = []
    model.to(device)

    for batch in dataset_test:

        inputs = batch["input_ids"].to(device=device, dtype=torch.long)
        mask = batch["attention_mask"].to(device=device)

        with torch.no_grad():
            output = model(input_ids=inputs, attention_mask=mask)

        logits = output.logits.detach().cpu()

        logits_flatten = (torch.sigmoid(logits).numpy()).tolist()

        y_pred_list.extend(logits_flatten)

    return y_pred_list


def get_responce_grnti_preds(preds, level=1, threshold=0.5,
                             decoding=True, dir_for_model="."):

    with open(f"{dir_for_model}/dict.json", "r") as code_file:
        grnti_mapping_dict_true_numbers = json.load(
            code_file
        )
        if decoding:
            with open(f"dicts/GRNTI_{level}_ru.json", "r",
                      encoding="utf-8") as name_file:
                grnti_mapping_dict_names_of_rubrics = json.load(name_file)

    list_GRNTI = []
    for el in preds:
        list_elments = {}

        for index, propab in enumerate(el):
            if propab >= threshold:
                list_elments[index] = propab
        list_GRNTI.append(list_elments)

    grnti_mapping_dict_true_numbers_reverse = {
        y: x for x, y in grnti_mapping_dict_true_numbers.items()
    }

    list_true_numbers_GRNTI = []
    for list_el in list_GRNTI:

        list_numbers = []
        for el in list_el:
            data_for_one_text = {}
            code_of_grnti = grnti_mapping_dict_true_numbers_reverse[el]

            data_for_one_text['code'] = code_of_grnti
            data_for_one_text['probability'] = list_el[el]
            if decoding:
                data_for_one_text['name'] =\
                      grnti_mapping_dict_names_of_rubrics[code_of_grnti]
            list_numbers.append(data_for_one_text)
        list_numbers = sorted(list_numbers,
                              key=lambda x: (-x["probability"], x['code']))
        list_true_numbers_GRNTI.append(list_numbers)

    return list_true_numbers_GRNTI
