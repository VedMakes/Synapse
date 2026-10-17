FROM python:3.11-slim

WORKDIR /app
COPY . /app
ENV PYTHON=python

RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --requirement requirements.lock

CMD ["make", "test"]
