import json


def load_global_xgbmodel(bst, globalModelPath):

    bst.load_model(globalModelPath)


def save_local_xgbmodel(bst, localModelPath):

    ckpt = {}
    ckpt["complete"] = json.loads(bst.save_raw("json").decode())
    if bst.num_boosted_rounds() == 1:
        ckpt["update"] = json.loads(bst.save_raw("json").decode())
    else:
        ckpt["update"] = json.loads(
            bst[bst.num_boosted_rounds() - 1 : bst.num_boosted_rounds()].save_raw("json").decode()
        )

    with open(localModelPath, "w") as outfile:
        json.dump(ckpt, outfile)
