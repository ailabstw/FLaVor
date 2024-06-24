FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
COPY . /app
workdir /app
RUN pip install -r requirements.txt

ENV PROCESS="python main.py"
CMD flavor-fl -m "${PROCESS}"
