import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    WALNUT_ENV: str = os.environ.get("WALNUT_ENV") or "development"
    DEVICE: str
    OCR_URL: str = "https://wise-ocr-4u55rnogeq-as.a.run.app"


@lru_cache()
def get_settings() -> Settings:
    environment = os.environ.get("WALNUT_ENV") or "development"
    return Settings(_env_file=f".env.{environment}")
