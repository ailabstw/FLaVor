# FLaVor Inference Service Overview

<p align="center">
    <img src="images/overview.png" width="400">
</p>

Welcome to the FLaVor Inference Service! This service simplifies the deployment of machine learning models by providing a user-friendly interface that encapsulates complex inference functions. It is designed to handle requests with various input formats seamlessly, ensuring that the inference outputs adhere to the structured AiCOCO format, a standard developed by Taiwan AILabs.

## How Does It Work

Using the FLaVor inference service is straightforward. Simply initiate `InferAPP` built upon a customized inference model. By sending a POST request to the `/invocations` endpoint of `InferAPP`, it will respond in a predefined format. The `/invocations` endpoint processes the request through the following stages:

## `/invocations` API Endpoint

- **Method**: POST
- **URL**: `/invocations`
- **Content-Type**: multipart/form-data
- **Body**: The request body should be formatted as multipart/form-data, allowing inclusion of multiple files and text fields.

  - **Required:**
    - `files`: The images for inference.
    - `data`: A JSON file with an `images` field in AiCOCO format, referencing the input image.
    - `metadata` (optional): Any additional information related to the files.

**Example**:

```python
import requests
import json

# Prepare data and files for the request
data = {
    "id": "0",
    "index": 0,
    "file_name": "0.dcm",
    "category_ids": None,
    "regressions": None
}

img_files = [("files", (f"images_image.jpg", open("image.jpg", "rb")))]
json_file["images"] = json.dumps(data)

# Send the POST request
response = requests.post(
    "http://0.0.0.0:9111/invocations",
    data=img_files,
    files=files,
)
```

Please refer to [`send_request.py`](send_request.py) for more detail.

## Getting Started

To begin with the FLaVor Inference Service, follow these steps:

### Step 1: Define Your Inference Model

Start by defining your custom inference model by subclassing the `BaseInferenceModel`. This serves as a template for implementing inference functionality tailored to your machine learning model.

Here's an example for a segmentation task. For a complete implementation, refer to the  [Segmentation task example](examples/inference/seg_example.ipynb).

```python
from flavor.serve.models import BaseAiCOCOInferenceModel

class SegmentationInferenceModel(BaseAiCOCOInferenceModel):
    ## Implement methods to define model-specific behavior
    ...

```

### Step 2: Customize Model-specific Behavior

<p align="left">
    <img src="images/call.png" width="700">
</p>

Override the abstract methods in your custom inference model to define the specific behavior of your model:

- `define_inference_network()`: Define the inference network or model and return a callable object or a network instance.
- `set_categories()`: Set inference categories and return `None` if no categories. For example, a segmentation output would contain `c` channels. By specifying in `set_categories()`, we show the exact meaning of each channel.
- `set_regressions()`: Set inference regressions and return `None` if no regressions. In segmentation task, here we simply return `None`.
- `data_reader()`: Read input data to numpy array or torch tensor.
- `output_formatter()`: Format the network output to a structured response. Currently four standard output strategy are available: `AiCOCOClassificationOutputStrategy`, `AiCOCODetectionOutputStrategy`, `AiCOCORegressionOutputStrategy`, and `AiCOCOSegmentationOutputStrategy`. See [Standard input and output structure](./docs/input_output_structure.md). In segmentation task, we choose `AiCOCOSegmentationOutputStrategy` as the formatter.

Besides, here are some non-abstract methods to override if necessary:

- `preprocess()`: Implement data transformation for the inference process.
- `inference()`: Implement forward operation.
- `postprocess()`: Implement any additional postprocessing steps for model output.

### Step 3: Run the Inference Service

Instantiate the `InferAPP` class from the FLaVor Inference Service, providing your `SegmentationInferenceModel` and required input and output data format. Then, run the application.

```python
from flavor.serve.apps import InferAPP
from flavor.serve.inference import BaseAiCOCOInputDataModel, BaseAiCOCOOutputDataModel

app = InferAPP(
    infer_function=SegmentationInferenceModel(),
    input_data_model=BaseAiCOCOInputDataModel,
    output_data_model=BaseAiCOCOOutputDataModel,
)
app.run(port=int(os.getenv("PORT", 9111)))

```

`InferAPP` serves as the central component of the FLaVor Inference Service, facilitating seamless interaction between other services and the machine learning models. To harness the power of `InferAPP`, developers need to provide the following essential components:

- `infer_function`: Developers can specify their custom inference model, allowing `InferAPP` to invoke the model and process its input/output seamlessly. Data reading, preprocessing, inference (network forward operation), postprocessing, and output formatting are performed accordingly. We also support Triton Inference Server to scale up the network forward operation. See the example in [SAM](./examples/inference/SAM)
- `input_data_model` and `output_data_model`: Developers can define the required Pydantic data model for the input request and output response.

### Step 4: Testing the Service by Sending Inference Requests

Once the FLaVor Inference Service is running, send POST requests to the `/invocations` endpoint with image data and associated JSON formatted in AiCOCO format to perform inference. See [`send_request.py`](send_request.py) for python example.

### More Examples for Various Tasks

Please visit following instruction pages:

- [Classification task example](./cls_example.ipynb)
- [Detection task example](./det_example.ipynb)
- [Regression task example](./reg_example.ipynb)
- [Segmentation task example](./seg_example.ipynb)
- [3D Segmentation task example](./seg3d_example.ipynb)

## Other Things You Might Want to Know

- [AiCOCO format specification](./docs/AiCOCO_spec.md)
- [Standard input and output structure](./docs/input_output_structure.md)
- [Visualize your inference output with Gradio](./gradio_example.ipynb)
