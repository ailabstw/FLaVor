# Detection task in FLaVor Inference Service

This guide will walk you through integrating the FLaVor inference service for 2D object detection task using the [YOLOv8-Medical-Imaging](https://github.com/sevdaimany/YOLOv8-Medical-Imaging) inference model.

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
# install package
# working directory: /your/path/FLaVor/examples/hello-inference
pip install -U https://github.com/ailabstw/FLaVor/archive/refs/heads/release/stable.zip && pip install "flavor[infer]"
pip install ultralytics
wget https://github.com/sevdaimany/YOLOv8-Medical-Imaging/raw/master/runs/detect/train/weights/best.pt
# initiate service
python main.py
```

### Docker Initiation

If you prefer Docker, you can build the environment using the provided [Dockerfile](./Dockerfile).

```bash
# working directory: /your/path/FLaVor/examples/hello-inference
# build docker image
docker build -t <your_image_name> -f dockerfile/Dockerfile.det .
# run the container
docker run -p 9999:9999 <your_image_name>
```

## Integration with InferAPP

The FLaVor Inference Service integrates an open-source inference model through `InferAPP`. The output of the inference model must be modified into a specific format for access by the output strategy, resulting in AiCOCO-formatted results.

```python
infer_output = {
    "sorted_images": [{"id": uid, "file_name": file_name, "index": index, ...}, ...],
    "categories": {class_id: {"name": category_name, "supercategory_name": supercategory_name, display: True, ...}, ...},
    "regressions": {regression_id: {"name": regression_name, "superregression_name": superregression_name, ...}, ...},
    "model_out": {
        "bbox_pred": bbox_pred, # list of bbox prediction as [[x_min, y_min, x_max, y_max], ...]
        "cls_pred": cls_pred, # list of 1d NumPy array as classification result of each bbox
        "confidence_score": confidence_score, # optional, list of the confidence values of the individual bbox
        "regression_value": regression_value, # optional, list of the regression value of each bbox if there is regression prediction
    }
}
```

Here, `model_out` must be a dictionary with keys as following:

* `bbox_pred`: A list of bounding box predictions as `[[x_min, y_min, x_max, y_max], ...]`, where each element should be of type `int`.
* `cls_pred`: A list of 1D NumPy arrays representing the classification result of each bounding box, where each element should be of type `int`.
* `confidence_score`: (Optional) A list of the confidence values of the individual bounding boxes, where each element should be of type `float`.
* `regression_value`: (Optional) A list of the regression values of each bounding box if there is regression prediction, where each element should be of type `float`.

## Testing example

Once the inference service is initiated, you can test it using the provided sample data and JSON file.

```bash
# working directory: /your/path/FLaVor/examples/hello-inference
python send_request.py -f test_data/det/BloodImage_00000_jpg.rf.5fb00ac1228969a39cee7cd6678ee704.jpg -d test_data/det/input.json
```

If everything runs smoothly, you should receive a response in the AiCOCO format.
