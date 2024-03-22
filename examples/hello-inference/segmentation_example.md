# Segmentation task in FLaVor Inference Service

This guide will walk you through integrating the FLaVor inference service for 2D segmentation tasks using the [lungmask](https://github.com/JoHof/lungmask) inference model.

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
pip install lungmask==0.2.18
# initiate service
python main.py
```

### Docker Initiation

If you prefer Docker, you can build the environment using the provided [Dockerfile](./Dockerfile).

```bash
# working directory: /your/path/FLaVor/examples/hello-inference
docker build -t <your_image_name> -f dockerfile/Dockerfile.seg .
# run the container
docker run -p 9999:9999 <your_image_name>
```

## Integration with InferAPP

The FLaVor Inference Service integrates an open-source inference model through `InferAPP`. The output of the inference model must be modified into a specific format for access by the output strategy, resulting in AiCOCO-formatted results.

```python
return_dict = {
    "sorted_images": [{"id": uid, "file_name": file_name, "index": index, ...}, ...],
    "categories": {class_id: {"name": category_name, "supercategory_name": supercategory_name, display: True, ...}, ...},
    "model_out": model_out # 3d/4d NumPy array with segmentation predictions
}
```

Here, `model_out` must be prediction masks representing by `0` or `1` for segmentation task. For instance segmentation tasks, `model_out` should only contain object masks with their unique IDs.

## Testing example

Once the inference service is initiated, you can test it using the provided sample data and JSON file.

```bash
# working directory: /your/path/FLaVor/examples/hello-inference
python send_request.py -f test_data/seg/0.dcm -d test_data/seg/input.json
```

If everything runs smoothly, you should receive a response in the AiCOCO format.
