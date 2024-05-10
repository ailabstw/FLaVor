FROM python:3.8.16-slim
COPY . /app
workdir /app
RUN pip install -r requirements.txt

ENV PROCESS="python main.py"
CMD flavor-fl -m "${PROCESS}"
