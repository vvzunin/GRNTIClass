from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
import json
from .prediction import prepair_model, prepair_dataset, make_predictions, \
    get_responce_grnti_preds
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
        # assuming text files; adjust if binary
        contents.append(content.decode())
        # important: rewind the file for potential future reads
        await file.seek(0)
    return contents


def load_device_config() -> torch.device:
    """Загружает устройство (CPU/GPU) из config.json."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            device_name = json.load(file)["device"]
        return torch.device(device_name)
    except (IOError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Ошибка загрузки конфига: {e}")


def send_sse_event(
    event_type: str,
    progress: Optional[int] = None,
    message: Optional[str] = None,
    filename: Optional[str] = None,
    rubrics: Optional[List] = None,
    completed: Optional[int] = None,
    total_files: Optional[int] = None
) -> str:
    """Генерирует SSE-событие в формате JSON."""
    event_data = {"type": event_type}
    if progress is not None:
        event_data["progress"] = progress
    if message:
        event_data["message"] = message
    if filename:
        event_data["filename"] = filename
    if rubrics:
        event_data["rubrics"] = rubrics
    if completed is not None and total_files is not None:
        event_data["completed"] = completed
        event_data["total_files"] = total_files
    return json.dumps(event_data) + "\n"


def prepare_grnti_levels(
    level1: bool,
    level2: bool,
    level3: bool
) -> List[Dict[str, Any]]:
    """Формирует список уровней ГРНТИ для обработки."""
    levels = []
    if level1:
        levels.append({
            "level": 1,
            "model_name": "./models/model1/bert_peft_level1",
            "n_classes": 36
        })
    if level2:
        levels.append({
            "level": 2,
            "model_name": "./models/model2/bert_peft_level2_with_labels_extra",
            "n_classes": 246
        })
    if level3:
        levels.append({
            "level": 3,
            "model_name": "./models/model3/bert_peft_level3_lora",
            "n_classes": 1265
        })
    return levels


async def process_single_level(
    model_info: Dict[str, Any],
    dataset_loader: Any,
    device: torch.device,
    threshold: float,
    decoding: bool
) -> List[List]:
    """Обрабатывает предсказания для одного уровня ГРНТИ."""
    model = prepair_model(
        n_classes=model_info['n_classes'],
        lora_model_path=model_info['model_name']
    )
    predictions = make_predictions(model, dataset_loader, device=device)
    return get_responce_grnti_preds(
        predictions,
        model_info['level'],
        threshold,
        decoding=decoding,
        dir_for_model=model_info['model_name']
    )


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
        yield send_sse_event(
            event_type="init",
            total_files=total_files,
            message=f"Начата обработка {total_files} файлов"
        )

        # 1. Подготовка уровней ГРНТИ
        list_levels = prepare_grnti_levels(level1, level2, level3)
        if not list_levels:
            yield send_sse_event(
                event_type="error",
                message="Не выбран уровень ГРНТИ"
            )
            return

        # 2. Загрузка конфига и данных
        yield send_sse_event(
            event_type="progress",
            progress=0,
            message="Подготовка данных",
            completed=0,
            total_files=total_files
        )

        try:
            files_texts = await get_files_content(files)
            dataset_loader = prepair_dataset(
                pd.DataFrame({"text": files_texts}))
            device = load_device_config()

            # 3. Обработка каждого уровня
            predictions_list = [[] for _ in range(total_files)]
            for index, model_info in enumerate(list_levels):
                progress = 10 + (90 / len(list_levels)) * index
                yield send_sse_event(
                    event_type="progress",
                    progress=progress,
                    message="Получение классов "
                    f"{model_info['level']}-го уровня",
                    completed=0,
                    total_files=total_files
                )

                level_predictions = await process_single_level(
                    model_info, dataset_loader, device, threshold, decoding
                )
                for i, preds in enumerate(level_predictions):
                    predictions_list[i].extend(preds)

            # 4. Отправка результатов
            for i, file in enumerate(files, 1):
                yield send_sse_event(
                    event_type="result",
                    filename=file.filename,
                    rubrics=predictions_list[i - 1]
                )
                yield send_sse_event(
                    event_type="file_complete",
                    completed=i,
                    total=total_files,
                    message=f"Обработан файл {i}/{total_files}"
                )

            yield send_sse_event(
                event_type="complete",
                message="Обработка завершена"
            )

        except Exception as e:
            yield send_sse_event(
                event_type="error",
                message=str(e)
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
