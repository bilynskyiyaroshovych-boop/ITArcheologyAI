FROM python:3.12-slim
WORKDIR /downloader
COPY requirements.txt /downloader/requirements.txt

RUN pip install --no-cache-dir -r /downloader/requirements-ml.txt

COPY . /downloader

CMD ["python", "downloader.py"]