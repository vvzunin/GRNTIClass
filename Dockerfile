FROM python:3.8

# WORKDIR /app

# Копируем backend и конфиг
COPY requirements.txt .
COPY requirements_pytorch.txt .
COPY src/backend/main.py .
COPY src/backend/GRNTI_*.json .
COPY src/backend/app ./app
COPY src/backend/bert_peft_level1_extra ./bert_peft_level1_extra
COPY src/backend/bert_peft_level2_with_labels_extra ./bert_peft_level2_with_labels_extra
COPY src/static/config.json ./static/config.json

RUN pip install --no-cache-dir -r requirements_pytorch.txt
RUN pip install --no-cache-dir -r requirements.txt

# Запускаем сервер (порт будет взят из config.json)
CMD ["python", "main.py"]