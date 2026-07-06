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
  "records": {
    // Required fields
    "format": "jsonl",
    "href": "/invocations/artifacts/records_<id>.jsonl",
    "rows": 1000,
    "bytes": 123456,
    "expires_at": null
  },
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

### Records Artifact

The `records.href` value is a downloadable JSONL artifact. Clients should issue a `GET` request to that href to stream per-record results. Each JSONL line has the same record structure previously embedded in the response:

```jsonl
{"id":"nanoid(21)","table_id":"nanoid(21)","row_indexes":[0],"category_ids":["nanoid(21)"],"regressions":null}
{"id":"nanoid(21)","table_id":"nanoid(21)","row_indexes":[1],"category_ids":null,"regressions":[{"regression_id":"nanoid(21)","value":130}]}
```
