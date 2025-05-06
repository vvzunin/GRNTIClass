#!/bin/bash

CONFIG_FILE="src/static/config.json"

# Проверяем существование файла конфига
if [ ! -f "$CONFIG_FILE" ]; then
  read -p "Ошибка: файл конфигурации $CONFIG_FILE не найден!"
  pause
  exit 1
fi

# Чтение параметров через Python
PORT=$(python -c "import json; f=open('$CONFIG_FILE'); data=json.load(f); print(data['api']['port']); f.close()")
DEVICE=$(python -c "import json; f=open('$CONFIG_FILE'); data=json.load(f); print(data['device']); f.close()")

# Проверяем, что порт не пустой
if [ -z "$PORT" ]; then
  read -p "Ошибка: порт не указан в config.json!"
  pause
  exit 
fi

echo "Запуск контейнера с параметрами:"
echo "Port: $PORT"

# Собираем образ
docker build -t my-backend --build-arg TORCH_VARIANT=$DEVICE . 

# Запускаем контейнер
docker run -d -p "$PORT:$PORT" my-backend

read -p "Контейнер успешно запущен в фоновом режиме"

pause  # Для Windows CMD