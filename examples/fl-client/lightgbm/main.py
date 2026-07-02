import os

import lightgbm as lgb
from sklearn.datasets import load_digits
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import train_test_split

from flavor.cook.utils import SaveInfoJson, SetEvent, WaitEvent
from flavor.gadget.lightgbm import load_global_lgbmodel, save_local_lgbmodel

total_rounds = int(os.getenv("TOTAL_ROUNDS"))

# Load example dataset from sklearn, please load from INPUT_PATH in the real-world scenario
data = load_digits()
X_train, X_val, y_train, y_val = train_test_split(
    data["data"], data["target"], test_size=0.2, random_state=0
)

# Training hyperparameter
param = {
    "objective": "multiclass",
    "num_class": 10,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "num_threads": 16,
    "verbose": -1,
}

# Tell the server that all preparations for training have been completed.
SetEvent("TrainInitDone")

bst = None
for round_idx in range(total_rounds):

    # Wait for the server
    WaitEvent("TrainStarted")

    dtrain = lgb.Dataset(X_train, y_train)

    if round_idx == 0:
        acc = 0.0

        # First round, add a single boosting iteration
        bst = lgb.train(param, dtrain, num_boost_round=1)
    else:
        # Load checkpoint sent from the server
        bst = load_global_lgbmodel(os.environ["GLOBAL_MODEL_PATH"])

        # Eval global model
        y_pred = bst.predict(X_val)
        acc = top_k_accuracy_score(y_val, y_pred, k=1)

        # Local train: continue one boosting iteration from the global model
        bst = lgb.train(param, dtrain, num_boost_round=1, init_model=bst)

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

    # Save newly added tree by slicing instead of the entire tree
    save_local_lgbmodel(bst, os.environ["LOCAL_MODEL_PATH"])

    # Tell the server that this round of training work has ended.
    SetEvent("TrainFinished")

# Notify the external service that the process is finished.
SetEvent("ProcessFinished")
