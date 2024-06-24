### Step 1: Get your aggregator code ready (for users who want to apply a custom aggregator)
 1. All required paths are stored in the `$OUTPUT_PATH` folder. Users can fetch them through `flavor.cook.utils.GetPaths`.
	 - `localModels`: All checkpoint paths sent from edges. (list, length=n_client)
	 - `localInfos`: All info.json paths returned from edges. (Servicer has already dealt with them, users can ignore)(list, length=n_client)
	 - `globalModel`: The path where the result of the aggregator needs to be saved.(list, length=1)
	 - `globalInfo`: The path where the merged global info.json needs to be saved. Note that dataset size for all edges are appended into a list and metrics are averaged. (Servicer has already dealt with it, users can ignore) (list, length=1)
 2. Users can choose to **re-initialize aggregator every round** or **only initialize at the beginning**, depending on how they implement. If choosing to initialize only at the beginning, two events, `AggregateStarted` and `AggregateFinished`, must be added to control the process.

- **Reminder**
  - If the training code is not implemented in Python, the user needs to implement several function imported from [`flavor.cook.utils`](../../flavor/cook/utils.py) in the example.
  - Disable all warnings. In Flavor, the environment variable `PYTHONWARNINGS` is already set to `ignore`, and `LOGLEVEL` (set to `ERROR`) is provided to the user. Or you can just add `SetEvent("ProcessFinished")` at the end of the code.

#### Code Example - Aggregator
```python
import json
from flavor.cook.utils import GetPaths, SetEvent, WaitEvent

class FedAvg(object):

    def aggregate(self, factors, localModelPaths):
        ......
        return globalModel

    def __call__(self):

        # Get Path
        localModelPaths = GetPaths("localModels")
        localInfoPaths = GetPaths("localInfos")
        globalModelPath = GetPaths("globalModel")[0]
        globalInfoPath = GetPaths("globalInfo")[0]

        # Caluculate aggregation factors
        with open(globalInfoPath, "r") as openfile:
            datasetSize = json.load(openfile)["metadata"]["datasetSize"]
        factors = [d / sum(datasetSize) for d in datasetSize]

        # Aggregate weights
        globalModel = self.aggregate(factors, localModelPaths)

        # Save model
        torch.save({"state_dict": globalModel}, globalModelPath)
```

#### Code Example - Initialize every round.
```python
def main():
    aggregator = FedAvg()
    aggregator()
```

#### Code Example - Initialize only once.
```python
def main():
    aggregator = FedAvg()
    while True:
        WaitEvent("AggregateStarted")
        aggregator()
        SetEvent("AggregateFinished")
```

### (Optional) Step 2:  Check implementation
Run `check-agg` to preliminarily check whether the implementation is correct on their computer before bundling the code into the Docker. To run `check-agg`, besides the code for the aggregator, the user also needs to prepare the training code for the client.
```bash
check-agg -m AGGREGATOR_PROCESS_CMD -cm CLIENT_MAIN_CMD -cp CLIENT_PREPROCESS_CMD(optional) -e NUM_OF_EPOCH(optional) -y(optional; automatic Enter to prompts) #Initialize every round.
```
or
```bash
check-agg --init-once -m AGGREGATOR_PROCESS_CMD -cm CLIENT_MAIN_CMD -cp CLIENT_PREPROCESS_CMD(optional) -e NUM_OF_EPOCH(optional) -y(optional; automatic Enter to prompts) #Initialize once.
```

### Step 3: Set Dockerfile CMD
Run aggregator through the following command:
```bash
flavor-agg -m AGGREGATOR_PROCESS_CMD #Initialize every round.
```
or
```bash
flavor-agg --init-once -m AGGREGATOR_PROCESS_CMD #Initialize once.
```
Bundle the code into the [Docker](Dockerfile) image and set `flavor-agg` as `CMD`.
```dockerfile
ENV PROCESS="python main.py"
CMD flavor-agg -m "${PROCESS}"
```

### Reminder
For more information about the AILabs FL Framework, including how to use the UI interface, please refer [here](https://harmonia.taimedimg.com/flp/documents/fl/2.0/manuals/).
