FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8 \
    && ln -sf /usr/bin/python3.8 /usr/bin/python \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential libssl-dev libffi-dev python3-dev python3-pip wget git\
    && apt-get install -y ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install https://github.com/ailabstw/FLaVor/archive/refs/heads/release/stable.zip -U && pip install "flavor[infer]"
RUN pip install tensorflow==2.8.0
RUN pip install protobuf==3.14.0

WORKDIR /app
COPY cls_example.py /app/

RUN git clone https://github.com/maxium0526/cft-chexpert.git /app/chexpert

CMD ["python", "main.py"]
