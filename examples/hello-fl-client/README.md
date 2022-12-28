### Step 1: Assign Process to EdgeAppServicer

Assign data preprocess (if any)  and training process to `EdgeAppServicer` as example. There are **5 additional arguments** that require passing into the training function.

 - **`namespace`**: Stores messages that need to be exchanged between the server and client.
	 - `globalModelPath`: [str] The model path sent back by the server.
	 - `epoch_path`: [str] The model path that the client is going to pass to the server in that round.
	 - `metadata`: [dict(float)] Contains meta information, i.e., "epoch" and "datasetSize".
	 - `metrics`: [dict(float)] Contains any metrics the user wants to monitor.
 - **`trainInitDoneEvent`**:  An event that tells the server that all preparations for training have been completed.
 - **`trainStartedEvent`**: The server uses this event to tell the client that it can continue with the next round of training. Therefore, the client needs to pause and wait before receiving the signal.
 - **`trainFinishedEvent`**: An event that tells the server that this round of training work has ended. After the server receives the signal, it will start aggregating the model weights received from each client.
 - **`logQueue`:** Store any messages to be sent to the server, including error handling. There are 3 types of logging level: `LogLevel.INFO`,`LogLevel.WARNING` and `LogLevel.ERROR`. When fatal errors happen, send a log with `LogLevel.ERROR`, and the systems will be shut down.

```python
from multiprocessing import Process
from flavors.cook.servicer import EdgeAppServicer

app_server = EdgeAppServicer()

app_server.dataPreProcess = None
app_server.trainingProcess = Process(
    target=MyTrainingFunction,
    kwargs={
         "namespace": app_server.namespace,
         "trainInitDoneEvent": app_server.trainInitDoneEvent,
         "trainStartedEvent": app_server.trainStartedEvent,
         "trainFinishedEvent": app_server.trainFinishedEvent,
         "logQueue": app_server.logQueue,
         ...... #Other self-defined arguments
     },
)
```

### Step 2: Modify the training code

After passing the namespace, event, and log queues to the training function, insert them in the corresponding places according to the definitions introduced in the previous step.

```python
import torch #Use PyTorch as an example
from flavors.cook.log_msg import PackLogMsg LogLevel #Include INFO, WARNING, and ERROR
import ... #Other packages you need

def run(namespace, trainInitDoneEvent, trainStartedEvent, trainFinishedEvent, logQueue, **kwargs):

	dataloader = ...
	optimizer = ...
	model = ...

	# Tell the server that all preparations for training have been completed.
	trainInitDoneEvent.set()
	logQueue.put(PackLogMsg(LogLevel.INFO, "Init Done")) #Not necessary

	for epoch in range(epochs):

		# Wait for the server
		trainStartedEvent.wait()
		trainStartedEvent.clear()

		# Load checkpoint sent from the server
		model.load_state_dict(torch.load(namespace.globalModelPath)["state_dict"])

		train()
		metrics = val() # Example: metrics = {"precision":0.8, "f1":0.7}

		# Save information that the server needs to know
		save_checkpoint(model, epoch_ckpt_path)
		namespace.metadata = {"epoch": epoch, datasetSize: len(dataset["train"])}
		namespace.metrics = metrics
		namespace.epoch_path = epoch_ckpt_path

		#Tell the server that this round of training work has ended.
		trainFinishedEvent.set()
```
```python
def MyTrainingFunction(namespace, trainInitDoneEvent, trainStartedEvent, trainFinishedEvent, logQueue, **kwargs):
	try:
        run(namespace, trainInitDoneEvent, trainStartedEvent, trainFinishedEvent, logQueue, **kwargs)
    except Exception as err:
	    #Error handling.
        logQueue.put(PackLogMsg(LogLevel.ERROR, str(err)))
```

### Step 3: Start Service.
```python
from flavors.cook.servicer import serve

serve(app_service)
```

### Reminder

 - Bundle code into a Docker image that can deploy to AILabs FL Framework.
 - More information can be found [here](https://harmonia.taimedimg.com/flp/documents/fl/2.0/manuals/).
