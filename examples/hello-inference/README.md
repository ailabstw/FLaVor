### Step 1: Wrap callback function with InferAPIApp


#### Code Example (pytorch)
```python
from flavor.serve.apps import InferAPP

# Toy infer function
def infer(**kwargs):
    return kwargs.get("images")

app = InferAPP(infer_function=infer,input_strategy=AiCOCOInputStrategy, output_strategy=AiCOCOutputStrategy)
app.run(port=9000)
```

#### Input Format (AiCOCOInputStrategy)
```json
{
    "images": [
        {
            // Required fields
            "id": "nanoid",
            "file_name": "<filename>.<ext>",
            "physical_file_name": "<physical_filename>.<ext>",
            // Optional fields
            "index": 1,
            // For time sequence images like US
            "category_ids": [
              "nanoid",
              "nanoid"
            ],
            ...
        }
    ]
}
```
#### Output Format (AiCOCOutputStrategy)

Output Tensor, AiCOCOutputStrategy converts the tensor to [aicoco](../../schema/ai-coco-v2.json). If output_strategy=None, your infer function must return aicoco directly.

##### AICOCO
```json
{
  "images": [
    {
      // Required fields
      "id": "nanoid",
      "file_name": "<filename>.<ext>",
      // Optional fields
      "index": 1,
      // For time sequence images like US
      "category_ids": [
        "nanoid",
        "nanoid"
      ],
      ...
    }
  ],
  "annotations": [
    {
      // Required fields
      "id": "nanoid",
      "image_id": "nanoid",
      "object_id": "nanoid",
      "iscrowd": 0,
      // 1 for RLE mask, 0 otherwise
      "bbox": [
        [
          x1,
          y1,
          x2,
          y2
        ],
        ...
      ]
      or
      null,
      "segmentation": [
        [
          x1,
          y1,
          x2,
          y2,
          x3,
          y3,
          ...,
          xn,
          yn
        ],
        ...
      ]
      or
      null,
      // Optional fields
      ...
    }
  ],
  "categories": [
    {
      // Required fields
      "id": "nanoid",
      "name": "Brain",
      "supercategory_id": "nanoid"
      or
      null,
      // Optional fields
      "color": "#FFFFFF",
      ...
    }
  ],
  "objects": [
    {
      // Required fields
      "id": "nanoid",
      "category_ids": [
        "nanoid",
        "nanoid"
      ],
      // Optional fields
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
    // Optional fields
    "task_type":  "binary" or "multiclass" or "multilabel",
    "category_ids": ["nanoid"]  // for whole series label
  }
}
```

### Step 2: Set Dockerfile CMD
Bundle the code into the Docker image and set `CMD`.
```dockerfile
CMD python main.py
```
