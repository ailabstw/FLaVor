### Step 1: Load data and model
Load data from from the environment variable `$INPUT_PATH` (folder) and model from `$WEIGHT_PATH` (file).

### Step 2: Export Results
Export the validation results to `{$OUTPUT_PATH}/result.json`. The file must contain two items, `metadata` and `results`, where `metadata` is the basic information of the edge, and `results` shows results and how users expect to present.
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
	* plots (line plot)
	  * title: string
	  * labels: string array
	  * x-values: number 2d array
	  * y-values: number 2d array
	  * x-axis: string
	  * y-axis: string
	* images (self-defined image file)
	  * title: string
	  * filename: string
#### Example
```json
  {
     "metadata":{
        "datasetSize":33
     },
     "results":{
        "tables":[
           {
              "title":"Overall Performance",
              "labels":["Accuracy"],
              "values":[0.7878]
           }
        ],
        "bars":[
           {
              "title":"Class 1 Bar Chart",
              "labels":["Precision", "Recall"],
              "values":[0.7619, 0.8889],
              "y-axis":"performance"
           }
        ],
        "heatmaps":[
           {
              "title":"Confusion Matrix",
              "x-labels":["Class0", "Class1"],
              "y-labels":["Class0", "Class1"],
              "values":[[10,5],[2,16]],
              "x-axis":"Prediction",
              "y-axis":"Ground truth"
           }
        ],
        "plots":[
           {
              "title":"ROC curve",
              "labels":["class 1"],
              "x-values":[[0, 0.2, 0.4, 0.8]],
              "y-values":[[0,0.25,0.43,0.83]],
              "x-axis":"True Positive rate",
              "y-axis":"False Positive rate"
           }
        ],
        "images":[
           {
              "title":"Example image",
              "filename":"example.png"
           }
        ]
     }
  }
```

### (Optional) Step 3:  Check implementation
Users may run `check-fv` to preliminarily check whether the implementation is correct on their computer before bundling the code into the Docker.
```bash
check-fv -m MAIN_PROCESS_CMD[required] -p PREPROCESS_CMD[optional]
```

### Step 4: Set Dockerfile CMD
Run federated validation through the following command:
```bash
flavor-fv -m MAIN_PROCESS_CMD -p PREPROCESS_CMD(optional)
```
Bundle the code into the [Docker](Dockerfile) image and set `flavor-fv` as `CMD`.
```dockerfile
ENV PROCESS="python main.py"
CMD flavor-fv -m "${PROCESS}"
```

### Reminder
For more information about the AILabs FV Framework, including how to use the UI interface, please refer [here](https://harmonia.taimedimg.com/flp/documents/fv/1.0/manuals/).
