# Classification task in FLaVor Inference Service

This guide will walk you through integrating the FLaVor inference service for 2D multi-label classification task using the [cft-chexpert](https://github.com/maxium0526/cft-chexpert) inference model.

## Prerequisite

Ensure you have the following dependencies installed:

```txt
python==3.8
torch==1.13.1+cu117
```

## Service Initiation

You can initiate the service either locally or using Docker.

### Local Initiation

```bash
# working directory: /your/path/FLaVor/examples/hello-inference
# install package
pip install -U https://github.com/ailabstw/FLaVor/archive/refs/heads/release/stable.zip && pip install "flavor[infer]"
git clone https://github.com/maxium0526/cft-chexpert.git chexpert
pip install tensorflow==2.8.0
pip install protobuf==3.14.0
# initiate service
python main.py
```

### Docker Initiation

If you prefer Docker, you can build the environment using the provided [Dockerfile](./Dockerfile).

```bash
# working directory: /your/path/FLaVor/examples/hello-inference
# build docker image
docker build -t <your_image_name> -f dockerfile/Dockerfile.cls .
# run the container
docker run -p 9999:9999 <your_image_name>
```

## Integration with InferAPP

The FLaVor Inference Service integrates an open-source inference model through `InferAPP`. The output of the inference model must be modified into a specific format for access by the output strategy, resulting in AiCOCO-formatted results.

```python
infer_output = {
    "sorted_images": [{"id": uid, "file_name": file_name, "index": index, ...}, ...],
    "categories": {class_id: {"name": category_name, "supercategory_name": supercategory_name, display: True, ...}, ...},
    "model_out": model_out # 1d NumPy array with classification predictions
}
```

Here, `model_out` must be prediction of each category (channel) representing by `0` or `1`. That means activation or thresholding should be performed beforehand.

## Testing example

Once the inference service is initiated, you can test it using the provided sample data and JSON file.

```bash
# working directory: /your/path/FLaVor/examples/hello-inference
python send_request.py -f chexpert/demo_img.jpg -d test_data/cls/input.json
```

If everything runs smoothly, you should receive a response in the AiCOCO format.
