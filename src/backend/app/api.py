from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, AsyncGenerator
import asyncio
import json
import time
import os

from .prediction import prepair_model, prepair_data_level1, prepair_data_level2, \
    prepair_dataset, make_predictions, save_rubrics, toRubrics
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

# async def process_file(file: UploadFile, level1: bool, level2: bool, level3: bool, threshold: float) -> AsyncGenerator[str, None]:
#     """Асинхронно обрабатывает один файл и генерирует события прогресса"""
#     try:
#         content = await file.read()
        
#         # Имитация обработки файла с прогрессом
#         # for progress in [20, 40, 60, 80, 100]:
#         #     await asyncio.sleep(0.5)  # Имитация работы
#             # yield json.dumps({
#             #     "type": "progress",
#             #     "filename": file.filename,
#             #     "progress": progress,
#             #     "message": f"Обработка файла {file.filename} ({progress}%)"
#             # }) + "\n"
        
#         # Генерация результатов
#         mock_results = await generate_mock_results(level1, level2, level3, threshold)
#         yield json.dumps({
#             "type": "result",
#             "filename": file.filename,
#             "rubrics": mock_results
#         }) + "\n"
#     except Exception as e:
#         yield json.dumps({
#             "type": "error",
#             "filename": file.filename,
#             "message": str(e)
#         }) + "\n"

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

# async def process_files_stream(files: List[UploadFile], level1: bool, level2: bool, level3: bool, threshold: float) -> AsyncGenerator[str, None]:
#     """Обрабатывает все файлы и генерирует поток событий"""
#     total_files = len(files)
    
#     # Начальное событие
#     yield json.dumps({
#         "type": "init",
#         "total_files": total_files,
#         "message": f"Начата обработка {total_files} файлов"
#     }) + "\n"
    
#     # Обработка каждого файла
#     for i, file in enumerate(files, 1):
#         async for chunk in process_file(file, level1, level2, level3, threshold):
#             yield chunk
        
#         # Событие завершения файла
#         yield json.dumps({
#             "type": "file_complete",
#             "completed": i,
#             "total": total_files,
#             "message": f"Завершена обработка файла {i} из {total_files}"
#         }) + "\n"

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
        
        # Отправляем начальное событие
        yield json.dumps({
            "type": "init",
            "total_files": total_files,
            "message": f"Начата обработка {total_files} файлов"
        }) + "\n"
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


        try:
            yield json.dumps({
                            "type": "progress",
                            "filename": "все файлы",
                            "progress": 0,
                            "message": f"Подготовка данных и моделей ({0}%)",
                            "completed": 1,
                            "total": total_files
                        }) + "\n"
            
            print("cwd = ", os.getcwd())
            if max_level ==1:
                modelType = 'bert_peft_level1_extra'
                model = prepair_model(n_classes=36, lora_model_path=modelType)
            else:
                modelType = 'bert_peft_level2_with_labels_extra'
                model = prepair_model(n_classes=246, lora_model_path=modelType)

            files_loc = ["Что-то по матиматике: уравнения", 
                    "Что-то по биологии: малекулы",
                    "Что-то по матиматике: алгебра", 
                    "Что-то по биологии: днк",
                    "Что-то по матиматике: прикладная математика", 
                    "Что-то по биологии: инфузория туфелька",
                    "Что-то по матиматике: дискретная математика", 
                    "Что-то по биологии: анатомия человека",
                    "Что-то по биологии: анатомия жука навозника"]
            dataset_loader = prepair_dataset(pd.DataFrame({"text":files_loc}))

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            yield json.dumps({
                        "type": "progress",
                        "filename": "все файлы",
                        "progress": 30,
                        "message": f"Получение предсказаний ({30}%)",
                        "completed": 1,
                        "total": total_files
                    }) + "\n"

            predictions = make_predictions(model, dataset_loader, device=device)

            predictions = toRubrics(predictions, 
                                        max_level-1 if max_level==3 else max_level, 
                                        threshold)
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
            
            for i, file in enumerate(files_loc, 1):
                # Генерация результатов
                mock_results = sorted_dicts_list_for_return[i-1]#await generate_mock_results(level1, level2, level3, threshold)
                print(f"{mock_results=}")
                yield json.dumps({
                    "type": "result",
                    "filename": f"Файл под номером {i}",
                    "rubrics": mock_results
                }) + "\n"
                
                # Событие завершения файла
                yield json.dumps({
                    "type": "file_complete",
                    "completed": i,
                    "total": total_files,
                    "message": f"Завершена обработка файла {i} из {len(files_loc)}"
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
