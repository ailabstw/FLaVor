### Step 1: Modify the training code
1. Handle Events
	- `TrainInitDone`: Tells the server that preparations for training have been completed.
	- `TrainStarted`: The server uses this event to tell the client that it can continue with the next round of training. Therefore, the client needs to pause and wait before receiving the signal.
	- `TrainFinished`: An event that tells the server that this round of training work has ended. After the server receives the signal, it will start aggregating the model weights received from each client.
2. Load dataset from the environment variable `$INPUT_PATH` (folder).
3. Load the global model from the environment variable `$GLOBAL_MODEL_PATH` (file) before each round of training starts. This step should be skipped in the first round if no pre-trained weights are provided.
4. Save the local model to `$LOCAL_MODEL_PATH` (file) after each round of training. If the default aggregator provided by AILabs is used, please save the weight in the format of `torch.Tensor` or `ndarray` to the ckpt/pickle file with dictionary key `state_dict`.
5. Export client information to `info.json` and save it in the same directory as the local model via `SaveInfoJson`. Please refer to the example below for the format.
- **Reminder**
  - If there are any log files, they can be saved to `$LOG_PATH` (folder); additional files can be put to`$OUTPUT_PATH` (folder).
  - If the training code is not implemented in Python, the user needs to implement several function imported from [`flavor.cook.utils`](../../flavor/cook/utils.py) in the example.

#### Code Example
```python
import os
import torch #Use PyTorch as an example
from flavor.cook.utils import SaveInfoJson, SetEvent, WaitEvent

def main():

    # Load the uploaded dataset
    dataloader = DataLoader(os.environ["INPUT_PATH"]...)
    optimizer = ...
    model = ...

    # (Handle Events) Tell the server that all preparations for training have been completed.
    SetEvent("TrainInitDone")

    for epoch in range(epochs):

        # (Handle Events) Wait for the server
        WaitEvent("TrainStarted")

        # Load checkpoint sent from the server (exclude epoch 0 if no pre-trained weight)
        model.load_state_dict(torch.load(os.environ["GLOBAL_MODEL_PATH"])["state_dict"])

        train()

        # Save checkpoint
        save_checkpoint({"state_dict": model.state_dict()}, os.environ["LOCAL_MODEL_PATH"])

        # To evaluate the global model, load again before the validation process (exclude epoch 0 if no pre-trained weight)
        model.load_state_dict(torch.load(os.environ["GLOBAL_MODEL_PATH"])["state_dict"])

        metrics = val() # Example: metrics = {"precision":0.8, "f1":0.7}

        # Save client information
        output = {}
        output["metadata"] = {"epoch": epoch,
                              "datasetSize": len(dataset["train"])}
                              "importance": importance
        output_dict["metrics"] = metrics
        SaveInfoJson(output_dict)

        # (Handle Events) Tell the server that this round of training work has ended.
        SetEvent("TrainFinished")
```

#### Json Example
```json
  {
     "metadata":{
        # (Required, int)
        "epoch":36,
        "datasetSize":100,
        # (Required, float) Importance factor for aggregator. If not using, just fill in a number.
        "importance":1.0
     },
     "metrics":{
        # (Required) If N/A or you don't want to track, fill in -1.
        "basic/confusion_tp":-1,
        "basic/confusion_fp":-1,
        "basic/confusion_fn":-1,
        "basic/confusion_tn":-1,

        # (Optional) Other metrics users expect to tracked.
        "mIOU":0.8500
     }
  }
```

### (Optional) Step 2:  Check implementation
Run `check-fl` to preliminarily check whether the implementation is correct on their computer before bundling the code into the Docker.
```bash
check-fl -m MAIN_PROCESS_CMD[required] -p PREPROCESS_CMD[optional]
```
If users are going to use their aggregator instead of the default one provided by AILabs, use `--ignore-ckpt` to skip the checkpoint checking step.
```bash
check-fl --ignore-ckpt -m MAIN_PROCESS_CMD[required] -p PREPROCESS_CMD[optional]
```

### Step 3: Set Dockerfile CMD
Run federated learning through the following command:
```bash
flavor-fl -m MAIN_PROCESS_CMD[required] -p PREPROCESS_CMD[optional]
```
Bundle the code into the [Docker](pytorch/Dockerfile) image and set `flavor-fl` as `CMD`.
```dockerfile
ENV PROCESS="python main.py"
CMD flavor-fl -m "${PROCESS}"
```

### Reminder
For more information including how to use the UI interface, please refer [here](https://harmonia.taimedimg.com/flp/documents/fl/2.0/manuals/).
