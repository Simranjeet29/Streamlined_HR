FROM python:3.9-slim
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
WORKDIR /app
EXPOSE 8080
ENTRYPOINT FLASK_APP=app.py flask run --host=0.0.0.0 --port=8080
