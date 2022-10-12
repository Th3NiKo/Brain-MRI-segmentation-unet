# Dockerfile for API server run
FROM python:3.10

WORKDIR /server

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000:8000

CMD [ "python", "./src/api/api.py"]
