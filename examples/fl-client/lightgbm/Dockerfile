FROM python:3.9.19-slim
COPY . /app
workdir /app
RUN pip install -r requirements.txt

ENV PROCESS="python main.py"
CMD flavor-fl -m "${PROCESS}"
