FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV SERVER_IP=172.17.0.1
ENV CLIENT_ID=1
CMD ["python", "main_client.py"]
