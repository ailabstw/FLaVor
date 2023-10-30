### Step 1: Wrap callback function with InferAPIApp


#### Code Example (pytorch)
```python
from flavor.serve.api import InferAPIApp
import EXAMPLE_MODEL

app = InferAPIApp(callback=EXAMPLE_MODEL)

app.run(port=int(os.getenv("PORT", 9000)))
```

#### Input Format
```json
{
	"input_path": ["<tmp_filename1>.<ext1>", "<tmp_filename2>.<ext2>"],
	"aicoco": {
		"images": [
	    {
	      // Required fields
	      "id": "uuidv4",
	      "file_name": "<filename>.<ext>",
	      // Optional fields
	      "index": 1,
	      // For time sequence images like US
	      "category_ids": [
	        "uuidv4",
	        "uuidv4"
	      ],
	      ...
	    }
		  ]
	}
}
```
#### Output Format ([schema/ai-coco-v2.json](../../schema/ai-coco-v2.json))
```json
{
  "images": [
    {
      // Required fields
      "id": "uuidv4",
      "file_name": "<filename>.<ext>",
      // Optional fields
      "index": 1,
      // For time sequence images like US
      "category_ids": [
        "uuidv4",
        "uuidv4"
      ],
      ...
    }
  ],
  "annotations": [
    {
      // Required fields
      "id": "uuidv4",
      "image_id": "uuidv4",
      "object_id": "uuidv4",
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
      "id": "uuidv4",
      "name": "Brain",
      "supercategory_id": "uuidv4"
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
      "id": "uuidv4",
      "category_ids": [
        "uuidv4",
        "uuidv4"
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
    "category_ids": ["uuidv4"]  // for whole series label
  }
}
```

### (Optional) Step 2:  Check implementation
Run `jsonschema.validate` on `schema/ai-coco-v2.json`


### Step 3: Set Dockerfile CMD
Bundle the code into the Docker image and set `CMD`.
```dockerfile
CMD python main.py
```
