# Standard Input and Output Structure (Tabular)

## Input
The FLaVor inference service accepts requests with form data containing input tabular data (.csv, .xls ...) and a JSON file in AiCOCO format.

The JSON file should adhere to the following structure:

```python
{
    "table": [
        # can be without order
        {
            "id": "nanoid",
            "index": index,
            "file_name": "<filename>.<ext>",
            ...
        },
        ...
    ],
    "meta": {
        "window_size": 1
    }
}
```

## Output
The output strategy includes functions that convert the inference model output to the AiCOCO format for serialization. This is typically used in the `output_formatter` of the inference model.

```python
from flavor.serve.inference.strategies import AiCOCOTabularClassificationOutputStrategy

...
def __init__(self,):
    self.formatter = AiCOCOTabularClassificationOutputStrategy()

def output_formatter(
        self,
        model_out: Any,
        tables: Sequence[AiTable],
        dataframes: Sequence[pd.DataFrame],
        meta: Dict[str, Any],
        categories: Optional[Sequence[Dict[str, Any]]] = None,
        regressions: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs,
    ) -> BaseAiCOCOTabularOutputDataModel:

    ...

    output = self.formatter(
                model_out=model_out,
                tables=tables,
                dataframes=dataframes,
                categories=categories,
                regressions=regressions,
                meta=meta
            )

    return output
```

Different output strategies can be selected based on the following task:

* [**Classification**](https://github.com/ailabstw/FLaVor/blob/fdf806dd574059fff2a03427596ea73814bfc5cc/flavor/serve/inference/strategies/aicoco_strategy.py#L808)  (binary, multiclass, multilabel)
* [**Regression**](https://github.com/ailabstw/FLaVor/blob/fdf806dd574059fff2a03427596ea73814bfc5cc/flavor/serve/inference/strategies/aicoco_strategy.py#L889) (single, multiple)


Each strategy uses specific input formats, defined as follows


## Classification

### **Usage**
Method: `__call__`

This method is the main function that applies the AiCOCO tabular output strategy to the model’s output.


```python
class AiCOCOTabularClassificationOutputStrategy(BaseAiCOCOTabularOutputStrategy):
    def __call__(
        self,
        model_out: np.ndarray,
        tables: Sequence[Dict[str, Any]],
        dataframes: Sequence[pd.DataFrame],
        categories: Sequence[Dict[str, Any]],
        meta: Dict[str, Any],
        **kwargs,
    ) -> AiCOCOTabularOut:
```

### Parameters
- `model_out`: np.ndarray

   2D NumPy array with classification results in shape of (n, c), where n represents n instances and c is the number of classes. Here are some output examples:

   > binary classification: [[0], [1], [1], ...]
   >
   > multiclass classification: [[0, 0, 1], [1, 0, 0], ...]
   >
   > multilabel classification: [[1, 0, 1], [1, 1, 1], ...]



- `tables`: Sequence[Dict[str, Any]]

   List of AiCOCO table compatible dict.

- `dataframes`: Sequence[pd.DataFrame]

   Sequence of dataframes correspond each tables.

- `categories`: Sequence[Dict[str, Any]]

   List of unprocessed categories.

- `meta`: Dict[str, Any]

   Additional metadata.

### Returns
- `AiCOCOTabularOut`: Dict[str, Any]

   A dictionary compatible with the full AiCOCO tabular format


## Regression

### **Usage**
Method: `__call__`

This method is the main function that applies the AiCOCO tabular output strategy to the model’s output.


```python
class AiCOCOTabularRegressionOutputStrategy(BaseAiCOCOTabularOutputStrategy):
    def __call__(
        self,
        model_out: np.ndarray,
        tables: Sequence[Dict[str, Any]],
        dataframes: Sequence[pd.DataFrame],
        regressions: Sequence[Dict[str, Any]],
        meta: Dict[str, Any],
        **kwargs,
    ) -> AiCOCOTabularOut:
```

### Parameters
- `model_out`: np.ndarray

   2D NumPy array with regression results in shape of (n, c), where n represents n instances and c is the number of regressions. Here are some output examples:

   > single regression: [[1], [2], [3], ...]
   >
   > multiple regression: [[1, 2, 3], [4, 5, 6], ...]

- `tables`: Sequence[Dict[str, Any]]

   List of AiCOCO table compatible dict.

- `dataframes`: Sequence[pd.DataFrame]

   Sequence of dataframes correspond each tables.

- `regressions`: Sequence[Dict[str, Any]]

   List of unprocessed regressions.

- `meta`: Dict[str, Any]

   Additional metadata.

### Returns
- `AiCOCOTabularOut`: Dict[str, Any]

   A dictionary compatible with the full AiCOCO tabular format


## Summary
The general pattern of the expected output should be a dictionary containing the following keys:

* `categories`:

   a dictionary where each key is the class ID in counting order starting from **0**.
   The corresponding value is a dictionary of category information that must be filled with name and optionally `supercategory_name`, `display` and all other details compatible in the AiCOCO format, except for the fields related to nanoid.

* `regressions`:

   a dictionary where each key is the regression ID in counting order starting from **0**.
   The corresponding value is a dictionary of regression information that must be filled with name and optionally `superregression_name` and all other details compatible in the AiCOCO format, except for the fields related to nanoid.

* `records`:

   a list of all predicted records that contains the following information
   * `row_indexes`: the indexes of the records in the raw datafram.
   * `category_ids`: the predicted categories corresponding to the records.
   * `regressions`: the predicted regression values corresponding to the records.
