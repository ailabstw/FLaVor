# Getting Started
`InferAPIApp` rationalizes the use of inference models by encapsulating the inference functions in a user-friendly framework. It provides easy-to-configure input and output strategies tailored to the processing and production of AiCOCO-formatted data, ensuring seamless integration and standardization.

## Step 1: Wrap Your Callback Function

Wrap the inference function of your model with the class `InferAPP`. Define the input strategy that converts the API input into the desired format, and optionally an output strategy that converts the model output into AiCOCO format.

### Code Example (PyTorch)

```python
from flavor.serve.apps import InferAPP
from flavor.serve.strategies import AiCOCOInputStrategy, AiCOCOutputStrategy

# Define your model's infer function
def infer(**kwargs):

    # Your model & inference code
    ...

    # please return your output as defined in Readme
    result = {
    "sorted_images": [...],
    "categories": {0: {"name": "Tumor", "supercategory_name": null, "display": False}},
    "seg_model_out": 4d ndarray with segmentation predictions
    }

    return result

# Wrap the infer function with input and optional output strategies
app = InferAPP(infer_function=infer,
               input_strategy=AiCOCOInputStrategy,
               output_strategy=AiCOCOutputStrategy()

# Run the application
app.run(port=9000)
```

### Input and Output Structures

#### AiCOCOInputStrategy

This strategy parses the incoming data into a structured format that the inference function can interpret.

##### Input Format Example

```python
{
    "images": [
        # may be without order
        {
            "id": "nanoid",
            "file_name": "<filename>.<ext>",
            ...
        },
        ...
    ]
}
```

#### AiCOCOOutputStrategy

The `AiCOCOutputStrategy` is an optional component that can be used to format the output of the inference model into the AiCOCO format, providing a standardized method for serializing the results. If `output_strategy=None` is specified, the `infer_function` must return the AiCOCO format directly.

#### Expected Model Output Structure for AiCOCOutputStrategy
```python
{
	"sorted_images": [
	    # z=0
	    {
	        "id": "nanoid",
	        "file_name": "<filename>.<ext>",
	        ...
	    },
	    # z=1
            {
	        "id": "nanoid",
	        "file_name": "<filename>.<ext>",
	        ...
	    },
	    ...
	]
	"categories": {
	    0: {"name": "Tumor", "supercategory_name": null, "display": False},
	    1: ...
	},

	"seg_model_out": 4d ndarray with segmentation predictions}
}
```
If  you  use  `AiCOCOutputStrategy`,  the  expected  output  should  be  a  dictionary  containing  the  following  keys:

 -  `sorted_images`:  A  list  of  images (see Input Format) sorted by  a certain criterion  (e.g.  by  Z-axis  or  temporal order) to  correlate  with  `seg_model_out`.

- `categories`: A dictionary where each key is the class ID. The corresponding value is a dictionary with category information that must be filled with `supercategory_name`, `display` and all necessary details as described in the AiCOCO format, except for fields related to "nanoid".

- `seg_model_out`: A 4D NumPy ndarray `(c, z, y, x)`, which represents the segmentation results. For semantic segmentation, the values are binary (0 or 1) and indicate the presence of a class. For instance segmentation, the array contains instance IDs as positive integers that indicate different instances.

- `det_model_out`: for bbox (To Do)

- `cls_model_out`: for classification (To Do)

##### AiCOCO Format

The AiCOCO format is described in detail below and contains the [schema](../../schema/ai-coco-v2.json)  for structured output, including required and optional fields for `images`, `annotations`, `categories`, and `objects`.

```python
{
  "images": [
    {
      # Required fields
      "id": "nanoid",
      "file_name": "<filename>.<ext>",
      # Optional fields
      "index": 1,
      # For time sequence images like US
      "category_ids": ["nanoid","nanoid"],
      ...
    }
  ],
  "annotations": [
    {
      # Required fields
      "id": "nanoid",
      "image_id": "nanoid",
      "object_id": "nanoid",
      "iscrowd": 0,
      # 1 for RLE mask, 0 otherwise
      "bbox": [[x1, y1, x2, y2],...] or null,
      "segmentation": [[x1, y1, x2, y2, x3, y3, ..., xn, yn],...] or null,
      # Optional fields
      ...
    }
  ],
  "categories": [
    {
      # Required fields
      "id": "nanoid",
      "name": "Brain",
      "supercategory_id": "nanoid" or null,
      # Optional fields
      "color": "#FFFFFF",
      ...
    }
  ],
  "objects": [
    {
      # Required fields
      "id": "nanoid",
      "category_ids": [
        "nanoid",
        "nanoid"
      ],
      # Optional fields
      "centroid": [
        x,
        y
      ],
      "confidence": 0.5,
      "regression_value": 50,
      ...
    }
  ],
  "meta": {
    # Optional fields
    "task_type":  "binary" # or "multiclass" or "multilabel",
    "category_ids": ["nanoid", ...]  # for whole series label
  }
}
```

Replace all placeholders with actual values, e.g.  `<filename>.<ext>`, `nanoid`, and details under each JSON key.

### Step 2: Configure the Dockerfile Command

Package your code in a Docker image and set the `CMD` directive to run your application.

#### Dockerfile Example

```dockerfile
CMD ["python", "main.py"]
```
