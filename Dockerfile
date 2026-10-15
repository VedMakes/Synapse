FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN python -m pip install --no-cache-dir --requirement requirements.lock

CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
