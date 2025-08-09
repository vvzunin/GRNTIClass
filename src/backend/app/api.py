from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
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
                    if torch.cuda.is_available() or torch.backends.mps.is_available()
                    else "cpu_only"
                ),
            },
        }

        model_paths = [
            "./models/model1/bert_peft_level1",
            "./models/model2/bert_peft_level2_with_labels_extra",
            "./models/model3/bert_peft_level3_lora",
        ]

        model_status = {}
        for i, path in enumerate(model_paths, 1):
            key = f"model{i}"
            model_status[key] = "available" if os.path.exists(path) else "not_found"
        health_status["models"] = model_status

        dict_status = {}
        for level in [1, 2, 3]:
            dict_path = os.path.join("dicts", f"GRNTI_{level}_ru.json")
            key = f"dict_level_{level}"
            dict_status[key] = "available" if os.path.exists(dict_path) else "not_found"
        health_status["dictionaries"] = dict_status

        return health_status

    except Exception as exc:
        return {
            "status": "unhealthy",
            "message": f"Backend error: {str(exc)}",
            "timestamp": pd.Timestamp.now().isoformat(),
        }


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
    Возвращает обычный JSON ответ.
    """
    start_time = time.time()
    try:
        total_files = len(files)
        print(f"Начало классификации {total_files} файлов")
        
        # Подготовка уровней ГРНТИ
        list_levels = []
        if level1:
            list_levels.append({
                "level": 1,
                "model_name": "./models/model1/bert_peft_level1",
                "n_classes": 36
            })
        if level2:
            list_levels.append({
                "level": 2,
                "model_name": "./models/model2/bert_peft_level2_with_labels_extra",
                "n_classes": 246
            })
        if level3:
            list_levels.append({
                "level": 3,
                "model_name": "./models/model3/bert_peft_level3_lora",
                "n_classes": 1265
            })
        
        if not list_levels:
            return {
                "type": "error",
                "message": "Не выбран уровень ГРНТИ"
            }

        print(f"Будет обработано уровней: {len(list_levels)}")

        # Читаем содержимое всех файлов
        files_texts = []
        files_names = []
        for file in files:
            try:
                await file.seek(0)
                content = await file.read()
                if not content:
                    raise ValueError("Файл пустой или не удалось прочитать содержимое")
                try:
                    decoded = content.decode("utf-8")
                except UnicodeDecodeError:
                    decoded = content.decode("cp1251", errors="replace")
                files_texts.append(decoded)
                files_names.append(file.filename)
                print(f"Прочитан файл: {file.filename}, размер: {len(content)} байт")
            except Exception as e:
                print(f"Ошибка при чтении файла {file.filename}: {e}")
                return {
                    "type": "error",
                    "message": f"Ошибка при чтении файла {file.filename}: {str(e)}"
                }

        if len(files_texts) == 0:
            return {
                "type": "error",
                "message": "Нет корректных файлов для обработки"
            }

        print(f"Подготовка данных для {len(files_texts)} файлов")
        # Подготовка данных для модели
        dataset_loader = prepair_dataset(pd.DataFrame({"text": files_texts}))

        # Загрузка конфигурации устройства
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                device_name = json.load(file)["device"]
        except IOError as e:
            print(f"Device name load error {e}")
            return {
                "type": "error",
                "message": f"Ошибка загрузки конфигурации: {str(e)}"
            }

        device = torch.device(device_name)
        print(f"Используется устройство: {device}")

        # Обработка каждого уровня ГРНТИ
        predictions_list = [[] for _ in range(len(files_texts))]

        for model_info in list_levels:
            level_start_time = time.time()
            print(f"Загрузка модели для уровня {model_info['level']}: {model_info['model_name']}")
            
            try:
                model = prepair_model(
                    n_classes=model_info["n_classes"],
                    lora_model_path=model_info["model_name"]
                )
                print(f"Модель уровня {model_info['level']} загружена за {time.time() - level_start_time:.2f}с")
                
                print(f"Выполнение предсказаний для уровня {model_info['level']}")
                pred_start_time = time.time()
                predictions = make_predictions(model, dataset_loader, device=device)
                print(f"Предсказания уровня {model_info['level']} выполнены за {time.time() - pred_start_time:.2f}с")
                
                print(f"Обработка результатов уровня {model_info['level']}")
                resp_start_time = time.time()
                predictions = get_responce_grnti_preds(
                    predictions,
                    model_info["level"],
                    threshold,
                    decoding=decoding,
                    dir_for_model=model_info["model_name"]
                )
                print(f"Результаты уровня {model_info['level']} обработаны за {time.time() - resp_start_time:.2f}с")
                
                for el_index, el_pred in enumerate(predictions):
                    predictions_list[el_index].extend(el_pred)
                
                # Очищаем память
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
                print(f"Уровень {model_info['level']} завершен за {time.time() - level_start_time:.2f}с")
                
            except Exception as e:
                print(f"Ошибка при обработке уровня {model_info['level']}: {e}")
                return {
                    "type": "error",
                    "message": f"Ошибка при обработке уровня {model_info['level']}: {str(e)}"
                }

        # Формирование результатов
        results = []
        for i, filename in enumerate(files_names):
            results.append({
                "filename": filename,
                "rubrics": predictions_list[i]
            })

        response_data = {
            "type": "result",
            "total_files": total_files,
            "results": results,
            "message": "Обработка завершена успешно",
            "processing_time": f"{time.time() - start_time:.2f}с"
        }
        
        print(f"Классификация завершена за {time.time() - start_time:.2f}с")
        return response_data

    except Exception as e:
        error_response = {
            "type": "error",
            "message": str(e),
            "processing_time": f"{time.time() - start_time:.2f}с"
        }
        print(f"Ошибка в API: {error_response}")
        return error_response