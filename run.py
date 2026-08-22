from __future__ import annotations

import argparse

from scripts.diagnose import collect


def main() -> int:
    parser = argparse.ArgumentParser(description="Video Character Extractor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--check", action="store_true", help="только проверить окружение")
    args = parser.parse_args()

    lines, errors = collect()
    print("\n".join(lines))
    if errors:
        print("\nЗапуск остановлен:")
        print("\n".join(f"- {item}" for item in errors))
        return 1
    if args.check:
        print("\nEnvironment is ready.")
        return 0

    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
