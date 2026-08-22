Python 3.10+

Локальная обработка:
- OpenCV
- Ultralytics
- YOLO26n
- TrackTrack
- PyTorch + MPS/Metal
- NumPy
- Pillow

Google API:
- Google AI Studio
- google-genai
- Gemini 3.1 Flash-Lite
  - дедупликация персонажей
  - выбор лучших референсов
  - анализ внешности и одежды
  - structured JSON
- Gemini 3.1 Flash Image
  - генерация ростового изображения по референсам

Backend:
- FastAPI
- Uvicorn
- python-multipart
- python-dotenv

Файлы проекта:
- requirements.txt
- requirements-dev.txt
- requirements-ci.txt
- run.py
- README.md

Пайплайн:
MP4
→ OpenCV
→ YOLO26n
→ TrackTrack + ReID
→ candidate crops
→ Gemini 3.1 Flash-Lite
→ unique characters + 4 best references
→ Gemini 3.1 Flash Image
→ full-body PNG
