# Standard Input and Output Structure

## About input structure

The FLaVor inference service accepts requests with form data containing input images and a JSON file in AiCOCO format. The JSON file should adhere to the following structure:

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

The output strategy includes functions that convert the inference model output to the AiCOCO format for serialization. This is typically used in the `output_formatter` of the inference model.

```python
from flavor.serve.inference.strategies import AiCOCOClassificationOutputStrategy

...
def __init__(self,):
    self.formatter = AiCOCOClassificationOutputStrategy()

def output_formatter(
    self,
    model_out: np.ndarray,
    images: Sequence[AiImage],
    categories: Sequence[Dict[str, Any]],
    **kwargs
) -> BaseAiCOCOImageOutputDataModel:

    output = self.formatter(model_out=model_out, images=images, categories=categories)
    return output
```

Different output strategies can be selected based on the task: classification, detection, regression, or segmentation. Each strategy uses specific input formats, defined as follows:


### Classification task

#### `class flavor.serve.inference.strategies.AiCOCOClassificationOutputStrategy`

```python
def __call__(
    model_out: np.ndarray,
    images: List[AiImage],
    categories: Sequence[Dict[str, Any]],
)
```

#### Arguments

* `model_out`: 1D NumPy array with classification results
* `images`: List of AiCOCO format compatible images
* `categories`: List of categories defined in your inference model (`set_categories`).

#### Return

* `AiCOCOImageOut`: A dictionary compatible with the full AiCOCO format

### Detection task (support 2D only)

#### `class flavor.serve.inference.strategies.AiCOCODetectionOutputStrategy`

```python
def __call__(
    model_out: Dict[str, Any],
    images: List[AiImage],
    categories: Sequence[Dict[str, Any]],
    regressions: Sequence[Dict[str, Any]],
):
```

#### Arguments

* `model_out`: Dictionary with predefined keys:
  * `bbox_pred`: List of bbox predictions as `[[x_min, y_min, x_max, y_max], ...]`
  * `cls_pred`: List of 1D NumPy arrays with classification results for each bbox
  * `confidence_score`: (Optional) List of confidence values for each bbox
  * `regression_value`: (Optional) List of regression values for each bbox
* `images`: List of AiCOCO format compatible images
* `categories`: List of categories defined in your inference model (`set_categories`).
* `regressions`: List of regressions defined in your inference model (`set_regressions`).

#### Return

* `AiCOCOImageOut`: A dictionary compatible with the full AiCOCO format

### Regression task

#### `class flavor.serve.inference.strategies.AiCOCORegressionOutputStrategy`

```python
def __call__(
    model_out: np.ndarray,
    images: List[AiImage],
    regressions: Sequence[Dict[str, Any]],
):
```

#### Arguments

* `model_out`: 1D NumPy array with regression results
* `images`: List of AiCOCO format compatible images
* `regressions`: List of regressions defined in your inference model (`set_regressions`).

#### Return

* `AiCOCOImageOut`: A dictionary compatible with the full AiCOCO format

### Segmentation task

#### `class flavor.serve.inference.strategies.AiCOCOSegmentationOutputStrategy`

```python
def __call__(
    model_out: np.ndarray,
    images: List[AiImage],
    categories: Sequence[Dict[str, Any]],
)
```

#### Arguments

* `model_out`: D NumPy array with segmentation results (instance or semantic segmentation mask)
* `images`: List of AiCOCO format compatible images
* `categories`: List of categories defined in your inference model (`set_categories`).

#### Return

* `AiCOCOImageOut`: A dictionary compatible with the full AiCOCO format

### Summary

The general pattern of the expected output should be a dictionary containing the following keys:

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
