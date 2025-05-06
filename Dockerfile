FROM python:3.8

# WORKDIR /app

# Копируем backend и конфиг
COPY requirements_docker.txt .
COPY src/backend/main.py .
COPY src/backend/GRNTI_*.json .
COPY src/backend/app ./app
COPY src/backend/bert_peft_level1_extra ./bert_peft_level1_extra
COPY src/backend/bert_peft_level2_with_labels_extra ./bert_peft_level2_with_labels_extra
COPY src/static/config.json ./static/config.json

ARG TORCH_VARIANT=cpu
RUN if [ "$TORCH_VARIANT" = "cuda" ]; then \
        PIP_URL="https://download.pytorch.org/whl/cu118"; \
        TORCH_PKG="torch==2.4.1+cu118"; \
    else \
        PIP_URL="https://download.pytorch.org/whl/cpu"; \
        TORCH_PKG="torch==2.4.1+cpu"; \
    fi; \
    pip install --index-url ${PIP_URL} ${TORCH_PKG}

RUN pip install --no-cache-dir -r requirements_docker.txt
RUN python -c "from transformers import AutoModelForSequenceClassification; AutoModelForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased')"
# Запускаем сервер (порт будет взят из config.json)
CMD ["python", "main.py"]