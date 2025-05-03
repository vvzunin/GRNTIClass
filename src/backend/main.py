from app.api import app
import uvicorn
import json
import os

if __name__ == "__main__":
    # Загружаем конфиг
    static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
    config_file_path = os.path.join(static_dir, "config.json")

    with open(config_file_path, "r") as f:
        config = json.load(f)

    docker_host = config["api"]["docker_host"]
    port = config["api"]["port"]

    uvicorn.run("app.api:app", host=docker_host, port=port, reload=True)