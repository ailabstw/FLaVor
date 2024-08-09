#!/usr/bin/env python
import argparse
import os
import sys

try:
    import torch
except ImportError:
    pass

import multiprocessing as mp
from platform import python_version

import numpy as np

from flavor.cook.utils import compareVersion

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LOGLEVEL"] = "ERROR"
os.environ["FLAVOR"] = "true"
os.environ["CHECK_FL"] = "true"


def transfer_data_to_tensor(batch):

    if isinstance(batch, (list, tuple, dict)):
        pass
    elif isinstance(batch, np.ndarray):
        return batch
    elif "torch" in sys.modules and isinstance(batch, torch.Tensor):
        return batch
    else:
        raise TypeError("Not supporting data type {}".format(type(batch)))

    # when list
    if isinstance(batch, list):
        for i, x in enumerate(batch):
            batch[i] = transfer_data_to_tensor(x)
        return batch

    # when tuple
    if isinstance(batch, tuple):
        batch = list(batch)
        for i, x in enumerate(batch):
            batch[i] = transfer_data_to_tensor(x)
        return tuple(batch)

    # when dict
    if isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = transfer_data_to_tensor(v)

        return batch

    return batch


def load_checkpoint(path):
    try:
        return torch.load(path)
    except Exception:
        import pickle

        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        return transfer_data_to_tensor(ckpt)


def set_env_var(key, default, force_default=False):
    if force_default:
        os.environ[key] = default
    else:
        os.environ[key] = (
            input(f"Set ${key} (Press ENTER if using default env - {default}): ") or default
        )


def run_app(app):

    states = ["data_validate", "train_init", "local_train", "train_finish"]

    for state in states:
        try:
            method = getattr(app, state)
            method({})
            if app.is_error.value != 0:
                return
        except Exception as err:
            app.close_process()
            app.AliveEvent.set()
            raise Exception(f"Error occurred: {err}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--main", type=str, required=True, help="main process command")
    parser.add_argument(
        "-p", "--preprocess", type=str, help="data preprocess command", default=None
    )
    parser.add_argument("-y", "--yes", action="store_true", help="skip setting env", default=False)
    parser.add_argument(
        "--ignore-ckpt",
        action="store_true",
        help="ignore ckpt check for 3rd party aggregator",
        default=False,
    )
    args, unparsed = parser.parse_known_args()

    force_default = args.yes
    set_env_var("INPUT_PATH", os.getenv("INPUT_PATH", "/data"), force_default)
    set_env_var("OUTPUT_PATH", os.getenv("OUTPUT_PATH", "/output"), force_default)
    set_env_var(
        "LOCAL_MODEL_PATH", os.getenv("LOCAL_MODEL_PATH", "/weight/local.ckpt"), force_default
    )
    set_env_var(
        "GLOBAL_MODEL_PATH", os.getenv("GLOBAL_MODEL_PATH", "/weight/global.ckpt"), force_default
    )
    set_env_var("LOG_PATH", os.getenv("LOG_PATH", "/log"), force_default)

    os.makedirs(os.environ["LOG_PATH"], exist_ok=True)
    os.makedirs(os.environ["OUTPUT_PATH"], exist_ok=True)
    os.makedirs(os.path.dirname(os.environ["LOCAL_MODEL_PATH"]), exist_ok=True)

    os.environ["TOTAL_ROUNDS"] = "1"

    from flavor.cook.app import EdgeApp

    app = EdgeApp(mainProcess=args.main, preProcess=args.preprocess, debugMode=True)

    server_process = mp.Process(target=run_app, args=(app,))
    server_process.start()

    app.AliveEvent.wait()

    server_process.terminate()
    server_process.join()
    if compareVersion(python_version(), "3.7") >= 0:
        server_process.close()

    if app.is_error.value != 0:
        raise Exception("Refer to ERROR log message")

    if not args.ignore_ckpt:
        print("Check Checkpoint")
        ckpt = load_checkpoint(os.environ["LOCAL_MODEL_PATH"])
        if "state_dict" not in ckpt:
            raise KeyError("state_dict not in checkpoint")
    else:
        if not os.path.exists(os.environ["LOCAL_MODEL_PATH"]):
            raise FileNotFoundError("Local model is not exist")
        print("Skip Checkpoint Checking")

    print("Run Successfullly !!!")


if __name__ == "__main__":

    main()
