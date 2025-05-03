from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import json
from .prediction import prepair_model, prepair_dataset, make_predictions, get_responce_grnti_preds
from tqdm import tqdm
import pandas as pd
import torch
import os

app = FastAPI()

# Настройки CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

        list_levels = []

        if level1:
            list_levels.append({"level": 1, "model_name": 'bert_peft_level1_extra',
                               "n_classes": 36})
        if level2:
            list_levels.append({"level": 2, 
                               "model_name": 'bert_peft_level2_with_labels_extra',
                               "n_classes": 246})
        if level3:# нужно будет имзенить для 3-го уровня
            list_levels.append({"level": 2, 
                               "model_name": 'bert_peft_level2_with_labels_extra',
                               "n_classes": 246})
        if not list_levels:
            yield json.dumps({
                "type": "error",
                "message": "Не выбран уровень ГРНТИ"
            }) + "\n"
            return  

        try:
            yield json.dumps({
                "type": "progress",
                "progress": 0,
                "message": "Подготовка данных",
                "completed": 0,
                "total_files": total_files
            }) + "\n"

            files_texts = await get_files_content(files)
            print(f"{files_texts=}")
    
            dataset_loader = prepair_dataset(pd.DataFrame({"text":files_texts}))

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            predictions_list =  [[] for i in range(total_files)]
            print(f"{list_levels=}")

            number_of_levels = len(list_levels)

            for index, model_info in enumerate(list_levels):
                progress = 10 + (90 / number_of_levels) * (index)
                yield json.dumps({
                    "type": "progress",
                    "progress": progress,
                    "message": f"Получение классов {model_info['level']}-го уровня ГРНТИ",
                    "completed": 0,
                    "total_files": total_files
                }) + "\n"

                model = prepair_model(n_classes= model_info['n_classes'] ,
                                    lora_model_path=model_info['model_name'])
                predictions = make_predictions(model, dataset_loader, device=device)

                predictions = get_responce_grnti_preds(predictions,
                                        model_info['level'], 
                                        threshold,
                                        decoding=decoding,
                                        dir_for_model=model_info['model_name'])
                for el_index, el_pred in enumerate(predictions):
                    predictions_list[el_index].extend(el_pred)

            for i, file in enumerate(files, 1):
                # Генерация результатов
                results = predictions_list[i-1]

                yield json.dumps({
                    "type": "result",
                    "filename": file.filename,
                    "rubrics": results
                }) + "\n"
                
                # Событие завершения файла
                yield json.dumps({
                    "type": "file_complete",
                    "completed": i,
                    "total": total_files,
                    "message": f"Завершена обработка файла {i} из {len(files)}"
                }) + "\n"

            yield json.dumps({
                "type":"progress",
                "filename": "все файлы",
                "progress": 100,
                "message": "Завершение обработки",
                "completed": total_files,
                "total": total_files
                }) + "\n"

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

# # Абсолютный путь к директории, которая находится выше
# static_dir = os.path.join(os.path.dirname(__file__), '../..', 'static')

# # Монтируем статическую директорию
# app.mount("/static", StaticFiles(directory=static_dir), name="static")

# @app.get("/config")
# async def get_config():
#     # Путь к файлу config.json относительно монтированной директории
#     config_file_path = os.path.join(static_dir, "config.json")
    
#     # Загружаем конфиг при запросе
#     with open(config_file_path, "r") as f:
#         config = json.load(f)
    
#     return config