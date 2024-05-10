import os

import xgboost as xgb
from sklearn.datasets import load_digits
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import train_test_split

from flavor.cook.utils import SaveInfoJson, SetEvent, WaitEvent
from flavor.gadget.xgboost import load_update_xgbmodel, save_update_xgbmodel

total_round = 10

# Load example dataset from sklearn, please load from INPUT_PATH in the real-world scenario
data = load_digits()
X_train, X_val, y_train, y_val = train_test_split(
    data["data"], data["target"], test_size=0.2, random_state=0
)

# Training hyperparmeter
param = {
    "objective": "multi:softprob",
    "eta": 0.1,
    "max_depth": 8,
    "nthread": 16,
    "num_parallel_tree": 1,
    "subsample": 1,
    "tree_method": "hist",
    "num_class": 10,
}

# Tell the server that all preparations for training have been completed.
SetEvent("TrainInitDone")

for round_idx in range(total_round):

    # Wait for the server
    WaitEvent("TrainStarted")

    # Reloading data to DMatrix are required if loading any model file due to xgb's bug
    dtrain = xgb.DMatrix(X_train, y_train)
    dval_X = xgb.DMatrix(X_val)

    if round_idx == 0:
        acc = 0.0

        # First round, set num_boost_round to 1
        bst = xgb.train(param, dtrain, num_boost_round=1)
    else:
        # Load checkpoint sent from the server
        load_update_xgbmodel(bst, os.environ["GLOBAL_MODEL_PATH"])

        # Save latest checkpoints
        bst.save_model(os.path.join(os.environ["OUTPUT_PATH"], "latest.json"))

        # Eval global model
        y_pred = bst.predict(dval_X)
        acc = top_k_accuracy_score(y_val, y_pred, k=1)

        # Local train
        bst.update(dtrain, bst.num_boosted_rounds())

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
    save_update_xgbmodel(bst, os.environ["LOCAL_MODEL_PATH"])

    # Tell the server that this round of training work has ended.
    SetEvent("TrainFinished")
