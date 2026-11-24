"""Start the app:  python -m app"""
import logging

import uvicorn

from app.config import get_env_settings

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_env_settings()
    print(f"\n  JobPilot AI -> http://{settings.host}:{settings.port}\n")
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, log_level="info")
