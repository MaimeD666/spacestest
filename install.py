from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(command: list[str]) -> None:
    print(f"\n> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Кроссплатформенная установка Video Character Extractor"
    )
    parser.add_argument(
        "--dev", action="store_true", help="также установить pytest/httpx"
    )
    parser.add_argument(
        "--skip-models", action="store_true", help="не загружать веса сейчас"
    )
    args = parser.parse_args()

    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        print("Нужен 64-bit Python 3.10, 3.11 или 3.12.", file=sys.stderr)
        return 2

    if not VENV_DIR.exists():
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    python = venv_python()
    if not python.is_file():
        print(f"Повреждённое окружение: не найден {python}", file=sys.stderr)
        return 2

    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    requirements = "requirements-dev.txt" if args.dev else "requirements.txt"
    run([str(python), "-m", "pip", "install", "-r", requirements])

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        shutil.copyfile(PROJECT_ROOT / ".env.example", env_path)
        print("\nСоздан .env. Добавьте в него GOOGLE_API_KEY.")
    else:
        print("\nСуществующий .env сохранён без изменений.")

    if not args.skip_models:
        run([str(python), "scripts/download_models.py"])

    if os.name == "nt":
        start_command = r".venv\Scripts\python.exe run.py"
    else:
        start_command = ".venv/bin/python run.py"
    print(f"\nУстановка завершена. После заполнения .env: {start_command}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
