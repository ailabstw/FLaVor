import os
import pickle

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from flavor.cook.utils import SaveInfoJson, SetEvent, WaitEvent

total_rounds = 2  # Must be 2
kernel = "rbf"

# Load example dataset from sklearn, please load from INPUT_PATH in the real-world scenario
data = load_digits()
X_train, X_val, y_train, y_val = train_test_split(
    data["data"], data["target"], test_size=0.2, random_state=0
)

# Tell the server that all preparations for training have been completed.
SetEvent("TrainInitDone")

for round_idx in range(total_rounds):

    # Wait for the server
    WaitEvent("TrainStarted")

    svm = SVC(kernel=kernel)

    if round_idx == 0:
        svm.fit(X_train, y_train)
        support_vec = {"support_x": X_train[svm.support_], "support_y": y_train[svm.support_]}
        acc = 0

    elif round_idx == 1:

        with open(os.environ["GLOBAL_MODEL_PATH"], "rb") as F:
            support_vec = pickle.load(F)
        svm.fit(support_vec["support_x"], support_vec["support_y"])
        y_pred = svm.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

    with open(os.environ["LOCAL_MODEL_PATH"], "wb") as F:
        pickle.dump(support_vec, F, protocol=pickle.HIGHEST_PROTOCOL)

    # Save information that the server needs to know
    output_dict = {}
    output_dict["metadata"] = {"epoch": round_idx, "datasetSize": len(X_train), "importance": 1.0}
    output_dict["metrics"] = {
        "accuracy": acc,
        "basic/confusion_tp": -1,  # If N/A or you don't want to track, fill in -1.
        "basic/confusion_fp": -1,
        "basic/confusion_fn": -1,
        "basic/confusion_tn": -1,
    }
    SaveInfoJson(output_dict)

    # Tell the server that this round of training work has ended.
    SetEvent("TrainFinished")

# Notify the external service that the process is finished.
SetEvent("ProcessFinished")
