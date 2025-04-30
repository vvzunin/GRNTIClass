from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, AsyncGenerator
import asyncio
import json
import time
import os

from .prediction import prepair_model, prepair_data_level1, prepair_data_level2, \
    prepair_dataset, make_predictions, toRubrics
from tqdm import tqdm
import pandas as pd
import torch


app = FastAPI()

# Настройки CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def generate_mock_results(level1: bool, level2: bool, level3: bool, threshold: float) -> List[dict]:
    """Генерирует моковые результаты с учетом максимального выбранного уровня"""
    await asyncio.sleep(0.1)  # Имитация обработки
    
    # Определяем максимальный выбранный уровень
    max_level = 0
    if level3:
        max_level = 3
    elif level2:
        max_level = 2
    elif level1:
        max_level = 1
    
    # if max_level
    
    mock_rubrics = [
        {"code": "01", "name": "Математика", "probability": 0.95},
        {"code": "01.03", "name": "Математика;Алгебра", "probability": 0.85},
        {"code": "01.03.05", "name": "Математика;Алгебра;Группы", "probability": 0.75},
        {"code": "02", "name": "Физика", "probability": 0.65},
        {"code": "02.01", "name": "Физика;Механика", "probability": 0.55},
        {"code": "02.01.03", "name": "Физика;Механика;Гидродинамика", "probability": 0.45},
        {"code": "03", "name": "Химия", "probability": 0.40},
        {"code": "03.02", "name": "Химия;Органическая", "probability": 0.35},
        {"code": "03.02.01", "name": "Химия;Органическая;Синтез", "probability": 0.30}
    ]
    
    filtered = []
    for rubric in mock_rubrics:
        # Определяем уровень текущей рубрики
        current_level = len(rubric["code"].split('.'))
        
        # Проверяем, что уровень рубрики не превышает максимальный выбранный
        if current_level <= max_level:
            if rubric["probability"] >= threshold:
                # Форматируем результат в соответствии с выбранным уровнем
                parts = rubric["name"].split(';')
                formatted_name = ';'.join(parts[:max_level])
                formatted_code = '.'.join(rubric["code"].split('.')[:max_level])
                
                filtered.append({
                    "code": formatted_code,
                    "name": formatted_name,
                    "probability": rubric["probability"],
                })
    
    # Сортируем по убыванию вероятности
    return sorted(filtered, key=lambda x: x["probability"], reverse=True)

async def get_files_content(files: List[UploadFile]) -> List[str]:
    contents = []
    for file in files:
        content = await file.read()
        contents.append(content.decode())  # assuming text files; adjust if binary
        await file.seek(0)  # important: rewind the file for potential future reads
    return contents


@app.post("/classify")
async def classify_files(
    files: List[UploadFile] = File(...),
    level1: bool = Form(True),
    level2: bool = Form(True),
    level3: bool = Form(True),
    normalization: bool = Form(True),
    decoding: bool = Form(True),
    threshold: float = Form(0.5)
) -> StreamingResponse:
    async def event_stream():
        total_files = len(files)
        print("files", files)
          # Имитация обработки
        print("До отправки начального сообщения")
        # Отправляем начальное событие
        yield json.dumps({
            "type": "init",
            "total_files": total_files,
            "message": f"Начата обработка {total_files} файлов"
        }) + "\n"
        print("После отправки начального сообщения")

        max_level = None

        if level3:
            max_level = 3
        elif level2:
            max_level = 2
        elif level1:
            max_level = 1

        if not max_level:
            yield json.dumps({
                "type": "error",
                "message": "Не выбран уровень ГРНТИ"
            }) + "\n"
            return  


        try:
            yield json.dumps({
                            "type": "progress",
                            "filename": "все файлы",
                            "progress": 0,
                            "message": f"Подготовка данных и моделей (0%)",
                            "completed": 1,
                            "total": total_files
                        }) + "\n"
            
            print("cwd = ", os.getcwd())
            if max_level ==1:
                modelType = 'bert_peft_level1_extra'
                n_classes = 36
            elif max_level==2:
                modelType = 'bert_peft_level2_with_labels_extra'
                n_classes = 246
            else:
                modelType = 'bert_peft_level3'
                n_classes = 1265

            model = prepair_model(n_classes=n_classes, lora_model_path=modelType)
            print("Начало работы с файлами")

            files_loc = await get_files_content(files)
            print(f"{files_loc=}")
    
            print("Конец работы с файлами")
    
            dataset_loader = prepair_dataset(pd.DataFrame({"text":files_loc}))

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print("До 'получение предсказиний'")
            yield json.dumps({
                        "type": "progress",
                        "filename": "все файлы",
                        "progress": 30,
                        "message": f"Получение предсказаний ({30}%)",
                        "completed": 1,
                        "total": total_files
                    }) + "\n"
            print("После 'получение предсказиний'")
            print("до make_predictions")

            predictions = make_predictions(model, dataset_loader, device=device)
            print("после make_predictions")

            predictions = toRubrics(predictions, max_level, 
                                        threshold)
            
            print("после toRubrics")

            sorted_dicts_list = [
                dict(sorted(dict_for_rubrick.items(),
                            reverse=True,
                            key=lambda x: (x[1][0] if x[1][0] else 0.) if x else 0.)) for
                                dict_for_rubrick in predictions]
            
            sorted_dicts_list_for_return = []

            for dict_for_rubrick in sorted_dicts_list:
                code = "Не определен"
                name = ""
                probability = None
                list_for_el = []
                if dict_for_rubrick:
                    for el, vals in dict_for_rubrick.items():
                        code = el
                        name = vals[1]
                        probability = vals[0]
                        list_for_el.append({"code":code,"name":name,
                                            "probability":probability})
                        # print(code, name, probability)

                    # {"code":code,"name":name,"probability":probability}
                # print("_______________________")
                sorted_dicts_list_for_return.append(list_for_el)
            print("Значения были получены")

       
            yield json.dumps({
                                "type": "progress",
                                "filename": "все файлы",
                                "progress": 100,
                                "message": f"Получены предсказания предсказаний (100%)",
                                "completed": 1,
                                "total": total_files
                            }) + "\n"
            
            for i, file in enumerate(files, 1):
                # Генерация результатов
                mock_results = sorted_dicts_list_for_return[i-1]#await generate_mock_results(level1, level2, level3, threshold)
                print(f"{mock_results=}")
                yield json.dumps({
                    "type": "result",
                    "filename": file.filename,
                    "rubrics": mock_results
                }) + "\n"
                
                # Событие завершения файла
                yield json.dumps({
                    "type": "file_complete",
                    "completed": i,
                    "total": total_files,
                    "message": f"Завершена обработка файла {i} из {len(files)}"
                }) + "\n"
            
            # Финальное событие
            yield json.dumps({
                "type": "complete",
                "message": "Обработка завершена"
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "message": str(e)
            }) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
