### Step 1: Assign Process to EdgeEvalServicer

Directly call data preprocess (if any)  and validation process as command line.
```python
from flavors.taste.servicer import EdgeEvalServicer

eval_service = EdgeEvalServicer()
eval_service.dataSubProcess = None
eval_service.valSubProcess = "python test.py"
```

### Step 2: Export Results

Export the validation results to `/output/result.json`. The file must contain two items, `metadata` and `results`, where `metadata` is the basic information of the edge, and `results` shows results and how users expect to present.
#### Format
 * metadata
	* datasetSize: integer
 * results
	* tables
	  * title: string
	  * labels: string array
	  * values: number array
	* bars
	  * title: string
	  * labels: string array
	  * values: number array
	  * y-axis: string
	* heatmaps
	  * title: string
	  * x-labels: string array
	  * y-labels: string array
	  * values: number 2d array
	  * x-axis: string
	  * y-axis: string
	* plots
	  * title: string
	  * labels: string array
	  * x-values: number 2d array
	  * y-values: number 2d array
	  * x-axis: string
	  * y-axis: string
	* images
	  * title: string
	  * filename: string
#### Example
```json
  {
    "metadata":{
      "datasetSize": 33
    },
    "results":{
      "tables": [
        {
          "title": "Overall Performance",
          "labels": ["Accuracy"],
          "values": [0.7878],
        },
      ],
      "bars": [
        {
          "title": "Class 1 Bar Chart",
          "labels": ["Precision", "Recall"],
          "values": [0.7619, 0.8889],
          "y-axis": "performance",
        },
      ],
      "heatmaps": [
        {
          "title": "Confusion Matrix",
          "x-labels": ["Class0","Class1"],
          "y-labels": ["Class0","Class1"],
          "values": [[10,5],[2,16]],
          "x-axis": "Predition",
          "y-axis": "Ground truth",
        },
      ],
      "plots": [
        {
          "title": "ROC curve",
          "labels": ["class 1"],
          "x-values": [[0,0.2,0,4,0,8]],
          "y-values": [[0,0.25,0.43,0.83]],
          "x-axis": "True Positive rate",
          "y-axis": "Faise Positive rate",
        }
      ],
      "images": [
        {
          "title": "Example image",
          "filename": "example.png"
        }
      ]
    }
  }
```

### Step 3: Start Service.
```python
eval_service.start()
```
