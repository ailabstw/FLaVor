### Step 1: Get your aggregator code ready
 1. All required paths are stored in the `$OUTPUT_PATH` folder. Users can fetch them through `GetPaths`.
	 - `localModels`: All checkpoint paths sent from edges.
	 - `localInfos`: All info.json paths returned from edges. (Servicer has already dealt with them, users can ignore)
	 - `globalModel`: The path where the result of the aggregator needs to be saved.
	 - `globalInfo`: The path where the merged global info.json needs to be saved. Note that dataset size for all edges are appended into a list and metrics are averaged. (Servicer has already dealt with it, users can ignore)
 2. Users can choose to **re-initialize aggregator every round** or **only initialize at the beginning**, depending on how they implement. If choosing to initialize only at the beginning, two events, `AggregateStarted` and `AggregateFinished`, must be added to control the process.

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

### Step 2: Set Dockerfile CMD
Run aggregator through the following command:
```bash
flavor-agg -m MAIN_PROCESS_CMD #Initialize every round.
```
or
```bash
flavor-agg --init-once -m MAIN_PROCESS_CMD #Initialize once.
```
Bundle the code into the [Docker](Dockerfile) image and set `flavor-agg` as `CMD`.
```dockerfile
ENV PROCESS="python main.py"
CMD flavor-agg -m "${PROCESS}"
```

### Reminder
For more information about the AILabs FL Framework, including how to use the UI interface, please refer [here](https://harmonia.taimedimg.com/flp/documents/fl/2.0/manuals/).
