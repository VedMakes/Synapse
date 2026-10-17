FROM python:3.11-slim

WORKDIR /app
COPY . /app
ENV PYTHON=python

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        make \
        libasound2 \
        libgl1 \
        libglib2.0-0 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libxrandr2 \
        libxcursor1 \
        libxi6 \
        libxinerama1 \
        libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --requirement requirements.lock

CMD ["make", "test"]
