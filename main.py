from app.gui import run_gui

if __name__ == "__main__":
    run_gui()

# python main.py

# docker build -t artifact-api .

# uvicorn api:app --host 0.0.0.0 --port 8000
# docker run -d -p 8000:8000 artifact-api
# ngrok http 8000

# https://preternaturally-incontestable-cristian.ngrok-free.dev/
# https://127.0.0.1:8000 