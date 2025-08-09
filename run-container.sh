#!/bin/bash

CONFIG_FILE="src/backend/config.json"

# Проверяем существование файла конфига
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Ошибка: файл конфигурации $CONFIG_FILE не найден!"
  read -p "Нажмите Enter для завершения..."
  exit 1
fi

# Чтение параметров через Python
PORT=$(python -c "import json; f=open('$CONFIG_FILE'); data=json.load(f); print(data['api']['port']); f.close()")
DEVICE=$(python -c "import json; f=open('$CONFIG_FILE'); data=json.load(f); print(data.get('device', 'cpu')); f.close()")

# Проверяем, что порт не пустой
if [ -z "$PORT" ]; then
  echo "Ошибка: порт не указан в config.json!"
  read -p "Нажмите Enter для завершения..."
  exit 1
fi

echo "========================================="
echo "Параметры из config.json:"
echo "Port: $PORT"
echo "Device: $DEVICE"
echo "========================================="

# Останавливаем и удаляем существующий контейнер, если он есть
echo "Проверяем существующий контейнер..."
if docker ps -a --format "table {{.Names}}" | grep -q "my-backend-container"; then
  echo "Останавливаем существующий контейнер..."
  docker stop my-backend-container
  docker rm my-backend-container
fi

# Определяем вариант PyTorch
if [ "$DEVICE" = "cuda" ] || [ "$DEVICE" = "gpu" ]; then
  TORCH_VARIANT="gpu"
  echo "Выбрана GPU версия PyTorch (CUDA)"
else
  TORCH_VARIANT="cpu"
  echo "Выбрана CPU версия PyTorch"
fi

echo "TORCH_VARIANT для Docker: $TORCH_VARIANT"
echo "========================================="

# Собираем образ
echo "Сборка Docker образа..."
docker build -t my-backend --build-arg TORCH_VARIANT=$TORCH_VARIANT .

# Проверяем успешность сборки
if [ $? -ne 0 ]; then
  echo "Ошибка при сборке Docker образа!"
  read -p "Нажмите Enter для завершения..."
  exit 1
fi

# Запускаем контейнер
echo "Запуск контейнера..."
docker run -d \
  --name my-backend-container \
  -p "$PORT:$PORT" \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/dicts:/app/dicts \
  my-backend

# Проверяем успешность запуска
if [ $? -ne 0 ]; then
  echo "Ошибка при запуске контейнера!"
  read -p "Нажмите Enter для завершения..."
  exit 1
fi

echo "Контейнер успешно запущен!"
echo "========================================="
echo "Логи контейнера:"
echo "========================================="

# Ждем немного и показываем логи
sleep 3
docker logs my-backend-container

echo "========================================="
echo "Для просмотра логов: docker logs my-backend-container"
echo "Для остановки: docker stop my-backend-container"
echo "Для удаления: docker rm my-backend-container"
echo "========================================="

# Проверяем статус контейнера
echo "Статус контейнера:"
docker ps --filter "name=my-backend-container"

echo "========================================="
read -p "Нажмите Enter для завершения..."