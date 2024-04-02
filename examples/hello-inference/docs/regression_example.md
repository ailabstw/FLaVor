# Regression task with the FLaVor Inference Service

This guide will walk you through integrating the FLaVor inference service for a 2D regression task using a dummy ResNet18 inference model.

## Prerequisite

Ensure you have the following dependencies installed:

```txt
python==3.8
torch==1.13.1+cu117
torchvision==0.14.1+cu117
```

## Service Initiation

You can initiate the service either locally or using Docker.

### Local Initiation

```bash
# working directory: /your/path/FLaVor/examples/hello-inference/regression_example
# install package
pip install -U https://github.com/ailabstw/FLaVor/archive/refs/heads/release/stable.zip && pip install "flavor[infer]"
# initiate service
python reg_example.py
```

### Docker Initiation

If you prefer Docker, you can build the environment using the provided [Dockerfile](examples/hello-inference/dockerfile/Dockerfile.reg).

```bash
# working directory: /your/path/FLaVor/examples/hello-inference
# build docker image
docker build -t <your_image_name> -f dockerfile/Dockerfile.reg .
# run the container
docker run -p 9000:9000 <your_image_name>
```

## Integration with InferAPP

The FLaVor Inference Service integrates an open-source inference model through `InferAPP`. The output of the inference model must be modified into a specific format for access by the output strategy, resulting in AiCOCO-formatted results.

```python
infer_output = {
    "sorted_images": [{"id": uid, "file_name": file_name, "index": index, ...}, ...],
    "regressions": {regression_id: {"name": regression_name, "superregression_name": superregression_name, ...}, ...},
    "model_out": model_out # 1d NumPy array with regression predictions
}
```

Here, `model_out` must be prediction of a series of regression values with `r` channels representing `r` individual results.

## Testing example

Once the inference service is initiated, you can test it using the provided sample data and JSON file.

```bash
# working directory: /your/path/FLaVor/examples/hello-inference
python send_request.py -f test_data/reg/test.jpeg -d test_data/reg/input.json
```

If everything runs smoothly, you should receive a response in the AiCOCO format.
