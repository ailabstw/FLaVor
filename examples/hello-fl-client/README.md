### Step 1: Modify the training code
 1. Handle events
	 - `TrainInitDone`: Tells the server that preparations for training have been completed.
	 - `TrainStarted`: The server uses this event to tell the client that it can continue with the next round of training. Therefore, the client needs to pause and wait before receiving the signal.
	 - `TrainFinished`: An event that tells the server that this round of training work has ended. After the server receives the signal, it will start aggregating the model weights received from each client.
 2. Load the global model from the environment variable `GLOBAL_MODEL_PATH` before each round of training. If no pre-trained weight is provided, skip this step at the initial round.
 3. Save the local model to the environment variable `LOCAL_MODEL_PATH` after each round of training.
 4. Save information that needs to be exchanged between the server and client via `SaveInfoJson`.

#### Code Example
```python
import os
import torch #Use PyTorch as an example
from flavors.cook.utils import SaveInfoJson, SetEvent, WaitEvent

def main():

	dataloader = ...
	optimizer = ...
	model = ...

	# Tell the server that all preparations for training have been completed.
	SetEvent("TrainInitDone")

	for epoch in range(epochs):

		# Wait for the server
		WaitEvent("TrainStarted")

		# Load checkpoint sent from the server (exclude epoch 0 if no pre-trained weight)
		model.load_state_dict(torch.load(os.environ["GLOBAL_MODEL_PATH"])["state_dict"])

		train()
		metrics = val() # Example: metrics = {"precision":0.8, "f1":0.7}

		# Save information that the server needs to know
		save_checkpoint({"state_dict": model.state_dict()}, os.environ["LOCAL_MODEL_PATH"])
		output_dict = {}
		output_dict["metadata"] = {"epoch": epoch, datasetSize: len(dataset["train"])}
		output_dict["metrics"] = metrics
		SaveInfoJson(output_dict)

        # Tell the server that this round of training work has ended.
        SetEvent("TrainFinished")
```

#### Json Example
```json
  {
    "metadata":{
      "datasetSize": 100,
      "epoch": 36
    },
    metrics:{
	  # (Required) If N/A or you don't want to track, fill in -1.
      "basic/confusion_tp": -1,
      "basic/confusion_fp": -1,
      "basic/confusion_fn": -1,
      "basic/confusion_tn": -1,

	  # (Optional) Other metrics users expect to tracked.
      "mIOU": 0.8500
	}

```
 **Reminder:** If the training code is not implemented in Python, the user needs to implement several function imported from `flavors.cook.utils` in the example.

### (Optional) Step 2:  Check implementation
After installing `flavors`, users may run `flavors-check` to preliminarily check whether the implementation is correct on their computer before bundling the code into the Docker . If pretrained weight is given, environment variables `GLOBAL_MODEL_PATH` must be set before executing the code.
```bash
export GLOBAL_MODEL_PATH=/ABC/DEF.ckpt(optional, if pretrained weight exists)
flavors-check -m MAIN_PROCESS_CMD[required] -p PREPROCESS_CMD[optional]
```
If users are going to use their aggregator instead of the default one provided by AILabs, use `--ignore-ckpt` to skip the checkpoint checking step.
```bash
flavors-check --ignore-ckpt -m MAIN_PROCESS_CMD[required] -p PREPROCESS_CMD[optional]
```

### Step 3: Set Dockerfile CMD
After installing `flavors`, users can run federated learning through the following command:
```bash
flavors-fl -m MAIN_PROCESS_CMD[required] -p PREPROCESS_CMD[optional]
```
Bundle the code into the Docker image and set `flavors-fl` as CMD.
```dockerfile
ENV PROCESS="python main.py"
CMD flavors-fl -m "${PROCESS}"
```


### Reminder
For more information about the AILabs FL Framework, including how to use the UI interface, please refer [here](https://harmonia.taimedimg.com/flp/documents/fl/2.0/manuals/).
