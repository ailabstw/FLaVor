# AiCOCO Format

AiCOCO format is devised by Taiwan AILabs to communicate across services.

## Specification example

The AiCOCO format is described in detail below. We also provide [schema](../../schema/aicoco.json) for structured output, including required and optional fields for `images`, `annotations`, `categories`, `regression` and `objects`. Please replace all placeholders with actual values, e.g.  `<filename>.<ext>`, `nanoid`, and details under each JSON key.

```python
{
  "images": [
    {
      # Required fields
      "id": "nanoid(21)",
      "file_name": "<filename>.<ext>",
      # For order purpose
      "index": 1,
      "category_ids": [
        "nanoid(21)", ...
      ] or null,
      "regressions": [
        {
          "regression_id": "nanoid(21)",
          "value": 130
        }
        , ...
      ] or null,
      #  Optional fields
      ...
    }
  ],
  "annotations": [
    {
      # Required fields
      "id": "nanoid(21)",
      "image_id": "nanoid(21)",
      "object_id": "nanoid(21)",
      "iscrowd": 0, # 1 for RLE mask, 0 otherwise
      "bbox": [
        [
          x1, y1, x2, y2
        ],
        ...
      ] or null,
      "segmentation": [
        [
          x1, y1, x2, y2, x3, y3, ..., xn, yn
        ],
        ...
      ] or null,
      # Optional fields
      ...
    }
  ],
  "categories": [
    {
      # Required fields
      "id": "nanoid(21)",
      "name": "Brain",
      "supercategory_id": "nanoid(21)" or null,
      # Optional fields
      "color": "#FFFFFF",
      ...
    }
  ],
  "regressions": [
    {
      # Required fields
      "id": "nanoid(21)",
      "name": "SBP",
      "superregression_id": "nanoid(21)" or null,
      # Optional fields
      "unit": "mmHg",
      "threshold": "140",
      ...
    }
  ],
  "objects": [
    {
      # Required fields
      "id": "nanoid(21)",
      "category_ids": [
        "nanoid(21)", ...
      ] or null,
      "regressions": [
        {
          "regression_id": "nanoid(21)",
          "value": 130
        }
        , ...
      ] or null,
      # Optional fields
      "confidence": 0.5,
      ...
    }
  ],
  "meta": {
    # Required
    # For whole series label
    "category_ids": ["nanoid(21)", ...] or null,
    "regressions": [
        {
          "regression_id": "nanoid(21)",
          "value": 130
        }
        , ...
      ] or null,
    # Optional fields
    "task_type":  "binary" or "multiclass" or "multilabel",
    ...
  }
}

```
