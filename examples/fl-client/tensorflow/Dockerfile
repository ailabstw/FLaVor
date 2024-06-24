FROM tensorflow/tensorflow:2.11.0-gpu
COPY . /app
workdir /app
RUN pip install -r requirements.txt

ENV PROCESS="python main.py"
CMD flavor-fl -m "${PROCESS}"
