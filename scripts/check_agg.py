#!/usr/bin/env python
import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from platform import python_version

from flavor.cook.model import AggregateRequest
from flavor.cook.utils import compareVersion

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LOGLEVEL"] = "ERROR"
os.environ["FLAVOR"] = "true"
os.environ["CHECK_AGG"] = "true"


def set_env_var(key, default, force_default=False):
    if force_default:
        os.environ[key] = default
    else:
        os.environ[key] = (
            input(f"Set ${key} (Press ENTER if using default env - {default}): ") or default
        )


def run_app(edge_app, aggregator_app, n_rounds):

    try:
        for state in ["data_validate", "train_init"]:
            getattr(edge_app, state)({})

            if edge_app.is_error.value != 0:
                return

        for round_idx in range(n_rounds + 1):

            getattr(edge_app, "local_train")({})
            if edge_app.is_error.value != 0:
                aggregator_app.close_process()
                return

            if round_idx == n_rounds:
                break

            shutil.move(
                os.environ["LOCAL_MODEL_PATH"],
                os.path.join(os.environ["REPO_ROOT"], "weights.ckpt"),
            )
            shutil.move(
                os.path.join(os.path.dirname(os.environ["LOCAL_MODEL_PATH"]), "info.json"),
                os.path.join(os.environ["REPO_ROOT"], "info.json"),
            )

            request = {"LocalModels": [{"path": ""}], "AggregatedModel": {"path": ""}}
            with open(
                os.path.join(os.path.dirname(os.environ["LOCAL_MODEL_PATH"]), "info.json")
            ) as json_file:
                info = json.load(json_file)
            request["LocalModels"][0].update(info)

            getattr(aggregator_app, "aggregate")(AggregateRequest(**request))
            if aggregator_app.is_error.value != 0:
                edge_app.close_process()
                return

            shutil.move(
                os.path.join(os.environ["REPO_ROOT"], "merged.ckpt"),
                os.environ["GLOBAL_MODEL_PATH"],
            )
            shutil.move(
                os.path.join(os.environ["REPO_ROOT"], "merged-info.json"),
                os.path.join(os.path.dirname(os.environ["GLOBAL_MODEL_PATH"]), "merged-info.json"),
            )

        getattr(edge_app, "train_finish")({})
        if edge_app.is_error.value != 0:
            aggregator_app.close_process()
            return

        getattr(aggregator_app, "train_finish")({})
        if aggregator_app.is_error.value != 0:
            edge_app.close_process()
            return

    except Exception as err:

        edge_app.close_process()
        aggregator_app.close_process()

        edge_app.is_error.value = 1
        edge_app.AliveEvent.set()

        raise Exception(f"Error occurred: {err}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--main", type=str, required=True, help="main process command")
    parser.add_argument("-r", "--rounds", type=int, default=1, help="round num to test")
    parser.add_argument(
        "--init-once", action="store_true", help="Initialize aggregator just once", default=False
    )
    parser.add_argument(
        "-cm", "--client-main", type=str, required=True, help="client main process command"
    )
    parser.add_argument(
        "-cp", "--client-preprocess", type=str, help="client data preprocess command", default=None
    )
    parser.add_argument("-y", "--yes", action="store_true", help="skip setting env", default=False)
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

    os.environ["REPO_ROOT"] = os.path.dirname(os.environ["LOCAL_MODEL_PATH"])

    os.makedirs(os.environ["LOG_PATH"], exist_ok=True)
    os.makedirs(os.environ["OUTPUT_PATH"], exist_ok=True)
    os.makedirs(os.path.dirname(os.environ["LOCAL_MODEL_PATH"]), exist_ok=True)

    if os.path.exists(os.environ["GLOBAL_MODEL_PATH"]):
        os.remove(os.environ["GLOBAL_MODEL_PATH"])
    os.makedirs(os.path.dirname(os.environ["GLOBAL_MODEL_PATH"]), exist_ok=True)

    os.environ["TOTAL_ROUNDS"] = args.rounds

    from flavor.cook.app import AggregatorApp, EdgeApp

    edge_app = EdgeApp(
        mainProcess=args.client_main, preProcess=args.client_preprocess, debugMode=True
    )
    aggregator_app = AggregatorApp(mainProcess=args.main, init_once=args.init_once, debugMode=True)

    server_process = mp.Process(
        target=run_app,
        args=(
            edge_app,
            aggregator_app,
            args.rounds,
        ),
    )
    server_process.start()

    while not (edge_app.AliveEvent.is_set() or aggregator_app.AliveEvent.is_set()):
        time.sleep(1)

    print("Wait for terminating ...")
    time.sleep(3)

    server_process.terminate()
    server_process.join()
    if compareVersion(python_version(), "3.7") >= 0:
        server_process.close()

    if edge_app.is_error.value != 0 or aggregator_app.is_error.value != 0:
        raise Exception("Refer to ERROR log message")

    print("Run Successfullly !!!")


if __name__ == "__main__":

    main()
