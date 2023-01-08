FROM nvcr.io/nvidia/pytorch:22.05-py3
COPY . /app
workdir /app
RUN pip install -r requirements.txt

ENV PROCESS="python main.py"
CMD flavors-fl -m "${PROCESS}"
