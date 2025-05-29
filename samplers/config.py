from pathlib import Path

CURRENT_FILE_PATH = Path(__file__).resolve()
REPOSITORY_DIRECTORY = CURRENT_FILE_PATH.parent.parent
MODELS_DIRECTORY = REPOSITORY_DIRECTORY / "models"
