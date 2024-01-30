FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8 \
    && ln -sf /usr/bin/python3.8 /usr/bin/python \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential libssl-dev libffi-dev python3-dev python3-pip wget \
    && apt-get install -y ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install https://github.com/ailabstw/FLaVor/archive/refs/tags/1.1.0b3.zip -U && pip install "flavor[infer]"

WORKDIR /app
COPY *.py /app/

CMD ["python", "main.py"]
