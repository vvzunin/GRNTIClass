from src.backend.app.api import app
import uvicorn
import json
import os

def backend_startup():
    # Загружаем конфиг
    # static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
    config_file_path = os.path.join(os.path.dirname(__file__), "config.json") # static_dir

    with open(config_file_path, "r") as f:
        config = json.load(f)

    # docker_host = config["api"]["docker_host"]
    local_host = config["api"]["local_host"]
    port = config["api"]["port"]

    uvicorn.run("src.backend.app.api:app", host=local_host, port=port, reload=True)
