FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8 \
    && ln -sf /usr/bin/python3.8 /usr/bin/python \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential libssl-dev libffi-dev python3-dev python3-pip wget\
    && apt-get install -y ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install https://github.com/ailabstw/FLaVor/archive/refs/heads/release/stable.zip -U && pip install "flavor[infer]"

RUN wget https://github.com/sevdaimany/YOLOv8-Medical-Imaging/blob/master/runs/detect/train/weights/best.pt
RUN wget https://github.com/sevdaimany/YOLOv8-Medical-Imaging/blob/master/DEMO_IMAGES/BloodImage_00000_jpg.rf.5fb00ac1228969a39cee7cd6678ee704.jpg -P test_data

WORKDIR /app
COPY detection_example/*.py /app
COPY detection_example/requirements.txt /app
COPY detection_example/test_data /app/test_data
COPY detection_example/best.pt /app
COPY send_request.py /app

RUN pip install -r requirements.txt

CMD ["python", "main.py"]
