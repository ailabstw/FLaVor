# Standard Input and Output Structure

## About input structure

The FLaVor inference service accepts requests with form data containing input images and a JSON file describing the images in AiCOCO format. The JSON file should adhere to the structure outlined below:

```python
{
    "images": [
        # can be without order
        {
            "id": "nanoid",
            "index": index,
            "file_name": "<filename>.<ext>",
            ...
        },
        ...
    ]
}
```

## About output structure

The output strategy is an optional component that allows the output of the inference model to be easily converted to the AiCOCO format and provides a standardized method for serializing the results. If `output_strategy=None` is specified, the `infer_function` must return the AiCOCO format directly.

The given output strategy can be chosen depending on the task, classification, detection, regression or segmentation. Each requires a different input format, which is defined in `AiCOCOClassificationOutputStrategy`, `AiCOCODetectionOutputStrategy`, `AiCOCORegressionOutputStrategy` and `AiCOCOSegmentationOutputStrategy`.

The input format for each output strategy are listed as follows:

### Classification task -  `AiCOCOClassificationOutputStrategy`

```python
infer_output = {
    "sorted_images": [{"id": uid, "file_name": file_name, "index": index, ...}, ...],
    "categories": {class_id: {"name": category_name, "supercategory_name": supercategory_name, display: True, ...}, ...},
    "model_out": model_out # 1d NumPy array with classification predictions
}
```

### Detection task (support 2D only) - `AiCOCODetectionOutputStrategy`

```python
infer_output = {
    "sorted_images": [{"id": uid, "file_name": file_name, "index": index, ...}, ...],
    "categories": {class_id: {"name": category_name, "supercategory_name": supercategory_name, display: True, ...}, ...},
    "regressions": {regression_id: {"name": regression_name, "superregression_name": superregression_name, ...}, ...},
    "model_out": {
        "bbox_pred": bbox_pred, # list of bbox prediction as [[x_min, y_min, x_max, y_max], ...]
        "cls_pred": cls_pred, # list of 1d NumPy array as classification result of each bbox
        "confidence_score": confidence_score, # optional, list of the confidence values of the individual bbox
        "regression_value": regression_value, # optional, list of the regression value of each bbox if there is a regression prediction
    }
}
```

### Regression task - `AiCOCORegressionOutputStrategy`

```python
infer_output = {
    "sorted_images": [{"id": uid, "file_name": file_name, "index": index, ...}, ...],
    "regressions": {regression_id: {"name": regression_name, "superregression_name": superregression_name, ...}, ...},
    "model_out": model_out # 1d NumPy array with regression predictions
}
```

### Segmentation task - `AiCOCOSegmentationOutputStrategy`

```python
infer_output = {
    "sorted_images": [{"id": uid, "file_name": file_name, "index": index, ...}, ...],
    "categories": {class_id: {"name": category_name, "supercategory_name": supercategory_name, display: True, ...}, ...},
    "model_out": model_out # 3d/4d NumPy array with segmentation predictions
}
```

### Summary

The general pattern of the expected output should be a dictionary containing the following keys:

* `sorted_images`: a list of AiCOCO-compatible images (see input format) sorted by a certain criterion (e.g. by Z-axis or temporal order) to correlate with `model_out`.

* `categories`: a dictionary where each key is the class ID in counting order starting from `0`. The corresponding value is a dictionary of category information that must be filled with `name` and optionally `supercategory_name`, `display` and all other details compatible in the AiCOCO format, except for the fields related to `nanoid`.

* `regressions`: a dictionary where each key is the regression ID in counting order starting from `0`. The corresponding value is a dictionary of regression information that must be filled with `name` and optionally `superregression_name` and all other details compatible in the AiCOCO format, except for the fields related to `nanoid".

* `model_out`:
  * Classification and regression tasks
    * **Output format**: The output is a 1-dimensional NumPy array with shape `(c,)`.
    * **Details**:
      * For tasks involving classification or regression, the result is displayed in this array.
      * In scenarios with multiple heads (multi-head cases), the outputs of each head should be concatenated. The concatenation order must follow the `category_ids`.

  * Detection task
    * **Output format**: The output for detection tasks is a dictionary with multiple key-value pairs.
    * **Details**:
      * `"bbox_pred"`: This key corresponds to a list of bounding box predictions. Each bounding box is represented as a list: `[x_min, y_min, x_max, y_max]`.
      * `"cls_pred"`: This key corresponds to a list of classification results for each bounding box. The structure is a list of lists, where the dimension of each inner list is `c`, which represents the category of each bounding box.
      * `"confidence_score"`: (Optional) This key contains a list of confidence values, one for each bounding box.
      * `"regression_value"`: (Optional) This is a list of regression values for each bounding box, structured as a list of lists similar to `cls_pred`, where each inner list refers to a bounding box.

  * Segmentation task
    * **Output format**: The output is either a 3-dimensional NumPy array with shape `(c, y, x)` or 4-dimensional array with shape `(c, z, y, x)`.
    * **Details**:
      * For semantic segmentation, the array values are binary (0 or 1) and indicate the presence or absence of a class.
      * For instance segmentation, the array contains positive integer values, each representing a unique instance ID to distinguish between different instances.
