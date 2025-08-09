FROM python:3.9

WORKDIR /app

# Копируем requirements для бэкенда
COPY requirements_backend.txt .
COPY requirements_ml.txt .

# Копируем исходный код бэкенда
COPY src/backend/ ./src/backend/
COPY src/start_backend.py ./src/start_backend.py

# Копируем данные и модели
COPY dicts/ ./dicts/
COPY models/ ./models/

# Устанавливаем базовые зависимости (без ML пакетов)
RUN pip install --no-cache-dir -r requirements_backend.txt

# Получаем параметр устройства (без значения по умолчанию)
ARG TORCH_VARIANT
RUN echo "Installing PyTorch for: $TORCH_VARIANT" && \
    if [ "$TORCH_VARIANT" = "gpu" ] || [ "$TORCH_VARIANT" = "cuda" ]; then \
        echo "Installing CUDA version of PyTorch..." && \
        pip install torch==2.7.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118; \
    else \
        echo "Installing CPU version of PyTorch..." && \
        pip install torch==2.7.0+cpu torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Устанавливаем ML пакеты которые зависят от PyTorch
RUN pip install --no-cache-dir -r requirements_ml.txt

# Предзагружаем модель для ускорения первого запуска
RUN python -c "from transformers import AutoModelForSequenceClassification; AutoModelForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased', use_safetensors=True)"

# Проверяем структуру файлов
RUN ls -la /app && \
    ls -la /app/src && \
    ls -la /app/src/backend && \
    ls -la /app/dicts && \
    ls -la /app/models

# Запускаем бэкенд из корневой директории проекта
CMD ["python", "src/start_backend.py"]