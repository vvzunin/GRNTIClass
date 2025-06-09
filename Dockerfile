FROM python:3.9

WORKDIR /work

# Копируем backend и конфиг
COPY requirements_docker.txt .

# Копируем исходный код
COPY src/backend/app src/backend/app
COPY src/backend/backend_startup.py src/backend/backend_startup.py
COPY src/backend/config.json src/backend/config.json

COPY src/backend/start_backend.py .

# Копируем данные и модели
COPY dicts/ dicts/
COPY models/ models/

# Указываем рабочую директорию для приложения
RUN ls -lR /work

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
RUN python -c "from transformers import AutoModelForSequenceClassification; AutoModelForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased', use_safetensors=True)"
CMD ["python", "start_backend.py"]