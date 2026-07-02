import json


def load_global_lgbmodel(globalModelPath):

    import lightgbm as lgb

    # The server writes the global booster text as a JSON string (see the
    # aggregator's FedLightGBM), mirroring the xgboost gadget's JSON wire format.
    with open(globalModelPath, "r") as infile:
        model_str = json.load(infile)

    return lgb.Booster(model_str=model_str)


def save_local_lgbmodel(bst, localModelPath):

    ckpt = {}
    ckpt["complete"] = bst.model_to_string()

    n_iter = bst.current_iteration()
    if n_iter == 1:
        ckpt["update"] = ckpt["complete"]
    else:
        ckpt["update"] = bst.model_to_string(start_iteration=n_iter - 1, num_iteration=1)

    with open(localModelPath, "w") as outfile:
        json.dump(ckpt, outfile)
