import json
import os
import uuid


def load_update_xgbmodel(bst, updateModelPath):

    with open(updateModelPath, "r") as json_file:
        updateModel = json.load(json_file)

    prevModel = json.loads(bst.save_raw("json").decode())

    update_num_trees = int(
        updateModel["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"]
    )
    prev_num_trees = int(
        prevModel["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"]
    )

    if prev_num_trees <= update_num_trees:
        bst.load_model(updateModelPath)
    else:
        prevModel["learner"]["gradient_booster"]["model"]["tree_info"][
            prev_num_trees - update_num_trees :
        ] = updateModel["learner"]["gradient_booster"]["model"]["tree_info"]
        update_trees = updateModel["learner"]["gradient_booster"]["model"]["trees"]
        for tree_idx in range(update_num_trees):
            update_trees[tree_idx]["id"] = prev_num_trees - update_num_trees + tree_idx
            prevModel["learner"]["gradient_booster"]["model"]["trees"][
                prev_num_trees - update_num_trees + tree_idx
            ] = update_trees[tree_idx]

        bst.load_model(bytearray(json.dumps(prevModel), "utf-8"))


def save_update_xgbmodel(bst, saveModelPath):
    # if not saveModelPath.lower().endswith(".json"):
    #    raise ValueError("Save path should be a json file. Got {}.".format(saveModelPath))

    _save_path = os.path.join(os.path.dirname(saveModelPath), str(uuid.uuid4()) + ".json")
    if bst.num_boosted_rounds() == 1:
        bst.save_model(_save_path)
    else:
        bst[bst.num_boosted_rounds() - 1 : bst.num_boosted_rounds()].save_model(_save_path)
    os.rename(_save_path, saveModelPath)
