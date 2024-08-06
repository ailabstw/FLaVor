# Tabular Model IO Specification

## POST `/invocations`

### Request

#### Header

```
content-type: multipart/form-data
```

#### Body

Target table files (whole dataset)

- **`"tables"`: `AiCOCO v2`.`tables`**
- **`"files"`: _Input files_**

### Response

#### Success

Response code: `200`

```python
{
  "result": AiCOCO v2
}
```

#### Failed

Response code: `400`, `500`

```python
{
  "error": "Error message"
}
```

#### Example

```python
{
  "tables": [
    {
      // Required fields
      "id": "nanoid(21)",
      "file_name": "<filename>.<ext>"
    }
  ],
  "categories": [
    {
      // Required fields
      "id": "nanoid(21)",
      "name": "Positive",
      "supercategory_id": "nanoid(21)" or null,
      // Optional fields
      "color": "#FFFFFF",
      ...
    }
  ],
  "regressions": [
    {
      // Required fields
      "id": "nanoid(21)",
      "name": "SBP",
      "superregression_id": "nanoid(21)" or null,
      // Optional fields
      "unit": "mmHg",
      "threshold": "140",
      ...
    }
  ],
  "records": [
    {
      // Required fields
      "id": "nanoid(21)",
      "table_id": "nanoid(21)",
      "row_indexes": [0, 1, ...],
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
      // Optional fields
      "confidence": 0.5,
      ...
    }
  ],
  "meta": {
    // Required fields
    "window_size": 1,
    // Optional fields
    "task_type":  "binary" or "multiclass" or "multilabel",
    "sample_id_column": "series_id",
    ...
  }
}
```
