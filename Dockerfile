FROM python:3.10-slim

# install system deps for torch & transformers if needed (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy code
COPY . /app
ENV PYTHONUNBUFFERED=1

# default command overridden in docker-compose
CMD ["python", "main_server.py"]
