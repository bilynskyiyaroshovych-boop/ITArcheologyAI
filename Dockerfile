FROM python:3.12-slim
WORKDIR /downloader


COPY requirements-ml.txt /downloader/requirements-ml.txt

RUN pip install --no-cache-dir --default-timeout=100 --retries 5 -r /downloader/requirements-ml.txt

COPY . /downloader

CMD ["/bin/bash"]