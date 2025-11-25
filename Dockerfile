FROM python:3.12-slim
WORKDIR /downloader
COPY requirements.txt /downloader/requirements.txt
# COPY .kaggle /root/.kaggle
RUN pip install --no-cache-dir -r /downloader/requirements.txt
COPY . /downloader
CMD ["python3", "downloader.py"]