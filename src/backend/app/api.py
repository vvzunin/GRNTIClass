from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertTokenizer
from typing import List
import json
import pandas as pd
import torch
import os
import time
import gc


from .prediction import (
    prepair_model,
    prepair_dataset,
    make_predictions,
    get_responce_grnti_preds,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


levels = [
    {
        "level": 1,
        "model_name": "./models/model1/bert_peft_level1",
        "n_classes": 36
    },
    {
        "level": 2,
        "model_name": "./models/model2/bert_peft_level2_with_labels_extra",
        "n_classes": 246
    },
    {
        "level": 3,
        "model_name": "./models/model3/bert_peft_level3_lora",
        "n_classes": 1265
    }
    ]


print("🚀 Начинаем загрузку моделей ")
print("=" * 60)

print("📦 Загружаем модель уровня 1...")
model_level_1 = prepair_model(
    n_classes=levels[0]["n_classes"],
    lora_model_path=levels[0]["model_name"]
)
print("✅ Модель уровня 1 успешно загружена")

print("📦 Загружаем модель уровня 2...")
model_level_2 = prepair_model(
    n_classes=levels[1]["n_classes"],
    lora_model_path=levels[1]["model_name"]
)
print("✅ Модель уровня 2 успешно загружена")

print("📦 Загружаем модель уровня 3...")
model_level_3 = prepair_model(
    n_classes=levels[2]["n_classes"],
    lora_model_path=levels[2]["model_name"]
)
print("✅ Модель уровня 3 успешно загружена")

print("🔤 Загружаем токенизатор...")
tokenizer = BertTokenizer.from_pretrained(
    "DeepPavlov/rubert-base-cased",
    do_lower_case=True
)
print("✅ Токенизатор успешно загружен")

print("=" * 60)
print("Все модели и токенизатор успешно загружены!")
print("=" * 60)


@app.get("/")
async def root():
    """Корневой эндпоинт для проверки доступности сервера."""
    return {"message": "GRNTI Classification API is running", "status": "ok"}


@app.get("/health")
async def health_check():
    """Эндпоинт для проверки состояния бэкенда."""
    try:
        health_status = {
            "status": "healthy",
            "message": "Backend is working correctly",
            "timestamp": pd.Timestamp.now().isoformat(),
            "components": {
                "api": "ok",
                "torch": (
                    "ok"
                    if torch.cuda.is_available() or
                    torch.backends.mps.is_available()
                    else "cpu_only"
                ),
            },
        }

        model_paths = [level["model_name"] for level in levels]

        model_status = {}
        for i, path in enumerate(model_paths, 1):
            key = f"model{i}"
            model_status[key] = "available" if os.path.exists(
                path) else "not_found"
        health_status["models"] = model_status

        dict_status = {}
        for level in [1, 2, 3]:
            dict_path = os.path.join("dicts", f"GRNTI_{level}_ru.json")
            key = f"dict_level_{level}"
            dict_status[key] = "available" if os.path.exists(
                dict_path) else "not_found"
        health_status["dictionaries"] = dict_status

        return health_status

    except Exception as exc:
        return {
            "status": "unhealthy",
            "message": f"Backend error: {str(exc)}",
            "timestamp": pd.Timestamp.now().isoformat(),
        }


async def _read_files(files):
    files_texts = []
    files_names = []
    for file in files:
        try:
            await file.seek(0)
            content = await file.read()
            if not content:
                raise ValueError(
                    "Файл пустой или не удалось прочитать содержимое")
            try:
                decoded = content.decode("utf-8")
            except UnicodeDecodeError:
                decoded = content.decode("cp1251", errors="replace")
            files_texts.append(decoded)
            files_names.append(file.filename)

        except Exception as e:
            return None, {
                "type": "error",
                "message": f"Ошибка при чтении файла {file.filename}: {str(e)}"
            }
    return (files_texts, files_names), None


def _get_device(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            device_name = json.load(file)["device"]
        device = torch.device(device_name)
        return device, None
    except IOError as e:
        return None, {
            "type": "error",
            "message": f"Ошибка загрузки конфигурации: {str(e)}"
        }


def _process_level(model_info, dataset_loader, device,
                   threshold, decoding):
    try:
        if model_info["level"] == 1:
            model = model_level_1
        elif model_info["level"] == 2:
            model = model_level_2
        elif model_info["level"] == 3:
            model = model_level_3

        predictions = make_predictions(model, dataset_loader, device=device)

        predictions = get_responce_grnti_preds(
            predictions,
            model_info["level"],
            threshold,
            decoding=decoding,
            dir_for_model=model_info["model_name"]
        )
        # Очищаем память
        del model
        (torch.cuda.empty_cache() if torch.cuda.is_available()
         else gc.collect())
        return predictions, None
    except Exception as e:
        return None, {
            "type": "error",
            "message": "Ошибка при обработке уровня"
            f" {model_info['level']}: {str(e)}"
        }


def _prepare_levels(level1, level2, level3):
    levels_to_process = []
    if level1:
        levels_to_process.append(levels[0])
    if level2:
        levels_to_process.append(levels[1])
    if level3:
        levels_to_process.append(levels[2])
    return levels_to_process


@app.post("/classify")
async def classify_files(
    files: List[UploadFile] = File(...),
    level1: bool = Form(True),
    level2: bool = Form(True),
    level3: bool = Form(True),
    decoding: bool = Form(True),
    threshold: float = Form(0.5),
):
    """
    Классификация файлов по уровням ГРНТИ.
    Возвращает JSON с результатами классификации.
    """
    start_time = time.time()
    try:
        total_files = len(files)
        list_levels = _prepare_levels(level1, level2, level3)
        if not list_levels:
            return {
                "type": "error",
                "message": "Не выбран уровень ГРНТИ"
            }
        # Читаем содержимое всех файлов
        (files_texts, files_names), file_error = await _read_files(files)
        if file_error:
            return file_error
        if not files_texts:
            return {
                "type": "error",
                "message": "Нет корректных файлов для обработки"
            }

        dataset_loader = prepair_dataset(
            pd.DataFrame({"text": files_texts}),
            tokenizer=tokenizer)

        # Загрузка конфигурации устройства
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..", "config.json")
        device, device_error = _get_device(config_path)
        if device_error:
            return device_error

        # Обработка каждого уровня ГРНТИ
        predictions_list = [[] for _ in range(len(files_texts))]
        for model_info in list_levels:
            predictions, level_error = _process_level(
                model_info, dataset_loader, device, threshold, decoding
            )
            if level_error:
                return level_error
            for el_index, el_pred in enumerate(predictions):
                predictions_list[el_index].extend(el_pred)

        # Формирование результатов
        results = [
            {"filename": filename, "rubrics": predictions_list[i]}
            for i, filename in enumerate(files_names)
        ]

        response_data = {
            "type": "result",
            "total_files": total_files,
            "results": results,
            "message": "Обработка завершена успешно",
            "processing_time": f"{time.time() - start_time:.2f}с"
        }

        return response_data

    except Exception as e:
        error_response = {
            "type": "error",
            "message": str(e),
            "processing_time": f"{time.time() - start_time:.2f}с"
        }
        return error_response
