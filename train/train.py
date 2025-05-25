from TrainSettings import TrainSettings
from datasets import Dataset
from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelAccuracy,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer
)
from peft import TaskType, LoraConfig, get_peft_model
import torch
import gc
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm as tqdm2
tqdm2.pandas()


def get_encoded_dataset(dataset, tokenizer,
                        max_length):
    """
    Кодирует датасет с использованием токенизатора.

    Args:
        dataset: Исходный датасет
        tokenizer: Токенизатор для обработки текста
        max_length: Максимальная длина токенизированного текста

    Returns:
        Токенизированный датасет в формате torch
    """
    def data_preprocesing(row):
        # Токенизация текста с ограничением длины
        return tokenizer(row['text'], truncation=True, max_length=max_length)

    tokenized_dataset = dataset.map(data_preprocesing, batched=True)

    tokenized_dataset.set_format("torch")
    return tokenized_dataset


def prepair_datasets(df, df_test, n_classes, level,
                     max_number_tokens=512,
                     pre_trained_model_name='DeepPavlov/rubert-base-cased'):
    """
    Подготавливает датасеты для обучения, валидации и тестирования.

    Args:
        df: Исходный датафрейм с данными
        df_test: Тестовый датафрейм
        n_classes: Количество классов
        level: Уровень кодирования целевых переменных
        max_number_tokens: Максимальное количество токенов
        pre_trained_model_name: Имя предобученной модели

    Returns:
        Кортеж из обучающего, валидационного и тестового датасетов,
        токенизатора, функции для создания батчей и весов классов
    """
    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(
        pre_trained_model_name, do_lower_case=True)

    # Подготовка разделения данных - убрали преобразование
    # в категории для списков
    target_series = df[f'target_coded{level}']

    # Подсчет встречаемости каждого уникального списка
    value_counts = Counter(tuple(x) for x in target_series)
    rare_classes = [k for k, v in value_counts.items() if v < 2]

    # Создаем маску для редких классов
    mask = target_series.apply(lambda x: tuple(x) in rare_classes)
    df_trunc_single = df[mask].copy()
    df_trunc_normal = df[~mask].copy()

    # Стратифицированное разделение - используем строковое представление
    # для стратификации
    train_df, valid_df = train_test_split(
        df_trunc_normal,
        stratify=df_trunc_normal[f'target_coded{level}'].astype(str),
        test_size=0.2
    )

    # Объединяем обучающий датасет с редкими классами
    train_df = pd.concat([train_df, df_trunc_single], ignore_index=True)

    # Эффективный по памяти расчет весов классов
    print("Calculating class weights...")
    class_counts = np.zeros(n_classes, dtype=np.int32)

    # Обработка целевых переменных по частям
    chunk_size = 10000
    for i in range(0, len(train_df), chunk_size):
        chunk = train_df[f'target_coded{level}'].iloc[i:i+chunk_size]
        class_counts += np.sum(np.vstack(chunk.values), axis=0)

    # Добавляем небольшую константу, чтобы избежать деления на 0
    class_counts = np.where(class_counts == 0, 1, class_counts)
    weights = torch.tensor(
        len(train_df) / (class_counts * n_classes), dtype=torch.float32)
    print("Class weights:", weights)

    # Создание датасетов
    def create_dataset(data):
        """
        Создает датасет из словаря текстов и меток.
        """
        return Dataset.from_dict({
            "text": data["text"].tolist(),
            "label": data[f'target_coded{level}'].tolist()
        })

    print("Preparing datasets...")
    datasets = {}
    for name, data in [("train", train_df),
                       ("valid", valid_df), ("test", df_test)]:
        print(f"Processing {name} data...")
        dataset = create_dataset(data)
        datasets[name] = get_encoded_dataset(
            dataset,
            tokenizer=tokenizer,
            max_length=max_number_tokens
        )
        del dataset
        gc.collect()

    # Создаем функцию для формирования батчей с паддингом
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    return (datasets["train"], datasets["valid"], datasets["test"],
            tokenizer, collate_fn, weights)


def prepair_model(n_classes,
                  pre_trained_model_name='DeepPavlov/rubert-base-cased',
                  r=16,
                  lora_alpha=32,
                  lora_dropout=0.05):
    """
    Подготавливает модель для многометочной классификации с настройкой LoRA.

    Args:
        n_classes: Количество классов
        pre_trained_model_name: Имя предобученной модели
        r: Ранг матриц для LoRA
        lora_alpha: Параметр масштабирования для LoRA
        lora_dropout: Вероятность дропаута для LoRA

    Returns:
        Модель с настроенным LoRA для дообучения
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        pre_trained_model_name,
        problem_type="multi_label_classification",
        num_labels=n_classes)
    print(model)

    # Настройка LoRA для модели
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["query", "key"],
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        # Модули, которые нужно сохранить полностью
        modules_to_save=['classifier', 'bert.pooler']
    )
    model_peft = get_peft_model(model, config)
    model_peft.print_trainable_parameters()

    return model_peft

# Функция подсчета всех метрик при валидации


def prepair_compute_metrics(n_classes):
    """
    Подготавливает функцию для вычисления метрик многометочной классификации.

    Args:
        n_classes: Количество классов

    Returns:
        Функция для вычисления метрик
    """

    multilabel_accuracy_micro = MultilabelAccuracy(
        num_labels=n_classes, average='micro')
    multilabel_accuracy_macro = MultilabelAccuracy(
        num_labels=n_classes, average='macro')
    multilabel_accuracy_weighted = MultilabelAccuracy(
        num_labels=n_classes, average='weighted')
    threshold_list = [0.5, 0.6, 0.7, 0.8, 0.9]

    # Создаем списки метрик F1 с разными порогами и типами усреднения
    multilabel_f1_score_micro_list = [MultilabelF1Score(
        num_labels=n_classes, average='micro',
        threshold=threshold) for threshold in threshold_list]
    multilabel_f1_score_macro_list = [MultilabelF1Score(
        num_labels=n_classes, average='macro',
        threshold=threshold) for threshold in threshold_list]
    multilabel_f1_score_weighted_list = [MultilabelF1Score(
        num_labels=n_classes, average='weighted',
        threshold=threshold) for threshold in threshold_list]

    def compute_metrics(pred):
        """
        Вычисляет метрики для предсказаний модели.

        Args:
            pred: Предсказания модели с метками

        Returns:
            Словарь с метриками
        """
        labels = torch.tensor(pred.label_ids).int()
        # Применяем сигмоиду для получения вероятностей
        preds = torch.sigmoid(torch.tensor(pred.predictions).float())

        # Вычисляем метрики точности с разными типами усреднения
        accuracy_micro = multilabel_accuracy_micro(preds, labels)
        accuracy_macro = multilabel_accuracy_macro(preds, labels)
        accuracy_weighted = multilabel_accuracy_weighted(preds, labels)

        # Вычисляем F1-меры с разными порогами и типами усреднения
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

        # Возвращаем словарь со всеми вычисленными метриками
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
            'f1_macro_0.9': f1_macro_09,
            'f1_weighted_0.9': f1_weighted_09,
        }
    return compute_metrics


class CustomTrainer(Trainer):
    """
    Кастомный класс тренера для обучения модели с весами классов.

    Расширяет стандартный Trainer из transformers
    для поддержки взвешенных потерь
    при мультиметочной классификации.
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            class_weights = class_weights.clone().detach().type(
                torch.float).to(self.args.device)
            self.class_weights = class_weights
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        """
        Вычисляет функцию потерь с учетом весов классов.

        Args:
            model: Модель для вычисления потерь
            inputs: Входные данные
            return_outputs: Флаг возврата выходов модели
            num_items_in_batch: Количество элементов в батче

        Returns:
            Значение функции потерь или кортеж (потери, выходы)
        """
        labels = inputs.pop("labels").to(
            self.args.device).float()  # Преобразуем метки в float

        outputs = model(**inputs)

        logits = outputs.get('logits')

        # Используем BCEWithLogitsLoss с весами или без
        if self.class_weights is not None:
            loss = torch.nn.BCEWithLogitsLoss(
                weight=self.class_weights).to(self.args.device)
        else:
            loss = torch.nn.BCEWithLogitsLoss().to(self.args.device)

        loss_res = loss(logits, labels)

        return (loss_res, outputs) if return_outputs else loss_res


def save_parameters(dir_name,
                    number_of_delteted_values,
                    minimal_number_of_elements_RGNTI2,
                    minimal_number_of_words, max_number_tokens,
                    pre_trained_model_name,
                    r, lora_alpha, lora_dropout, epoch, batch_size,
                    weight_decay, warmup_steps,
                    fp16, optim):
    """
    Сохраняет параметры эксперимента в файл настроек.

    Args:
        dir_name: Директория для сохранения
        number_of_delteted_values: Количество удаленных значений
        minimal_number_of_elements_RGNTI2: Минимальное количество
        элементов RGNTI2
        minimal_number_of_words: Минимальное количество слов
        max_number_tokens: Максимальное количество токенов
        pre_trained_model_name: Имя предобученной модели
        r: Ранг для LoRA
        lora_alpha: Параметр масштабирования для LoRA
        lora_dropout: Вероятность дропаута для LoRA
        epoch: Количество эпох обучения
        batch_size: Размер батча
        weight_decay: Регуляризация весов
        warmup_steps: Шаги прогрева оптимизатора
        fp16: Флаг использования половинной точности
        optim: Оптимизатор
    """

    sett = TrainSettings()
    sett.settings["number_of_delteted_values"] = number_of_delteted_values
    sett.settings["minimal_number_of_elements_RGNTI2"] =\
        minimal_number_of_elements_RGNTI2
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
    sett.save(path=dir_name)
