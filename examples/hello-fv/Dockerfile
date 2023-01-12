FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
COPY . /app
workdir /app
RUN pip install -r requirements.txt

ENV PROCESS="python main.py"
CMD flavors-fv -m "${PROCESS}"
