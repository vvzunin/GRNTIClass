import uvicorn
import json
import os


def backend_startup():
    # Загружаем конфиг
    config_file_path = os.path.join(os.path.dirname(__file__), "config.json")

    with open(config_file_path, "r") as f:
        config = json.load(f)

    local_host = config["api"]["host"]
    port = config["api"]["port"]

    uvicorn.run("backend.app.api:app",
                host=local_host, port=port, reload=False)
