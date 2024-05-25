import json
import logging
import multiprocessing as mp
import os
import subprocess
import time
from platform import python_version

import requests
import uvicorn
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from jsonschema import validate

from .model import AggregateRequest, LocalTrainRequest
from .utils import (
    CleanAllEvent,
    CleanEvent,
    CleanInfoJson,
    IsSetEvent,
    SaveGlobalInfoJson,
    SetEvent,
    SetPaths,
    WaitEvent,
    compareVersion,
)

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LOGLEVEL"] = "ERROR"
os.environ["FLAVOR"] = "true"

APPLICATION_REST_API_URI = os.getenv("APPLICATION_REST_API_URI", "0.0.0.0:8081")
OPERATOR_REST_API_URI = os.getenv("OPERATOR_REST_API_URI", "127.0.0.1:8080")


log_format = logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(log_format)

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
logger_path = os.path.join(os.environ["LOG_PATH"], "appService.log")
handler = logging.FileHandler(logger_path, mode="w", encoding=None, delay=False)
handler.setFormatter(log_format)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(consoleHandler)

process_logger = logging.getLogger("process")
process_logger.setLevel(logging.INFO)
process_logger_path = os.path.join(os.environ["LOG_PATH"], "appProcess.log")
process_handler = logging.FileHandler(process_logger_path, mode="w", encoding=None, delay=False)
process_handler.setFormatter(log_format)
process_handler.setLevel(logging.INFO)
process_logger.addHandler(process_handler)
process_logger.addHandler(consoleHandler)


class BaseAPP(object):
    def __init__(self, mainProcess, preProcess=None, debugMode=False):

        self.app = FastAPI()

        self.app.add_api_route(
            "/ping",
            self.ping,
            methods=["get"],
        )

        self.preProcess = preProcess
        self.mainProcess = mainProcess
        self.monitorProcess = None
        self.p = None

        self.AliveEvent = mp.Event()

        self.debugMode = debugMode

    def ping(self):
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    def run(self):
        logger.info(f"[run] Run app... {APPLICATION_REST_API_URI}")
        host = APPLICATION_REST_API_URI.split(":")[0]
        port = int(APPLICATION_REST_API_URI.split(":")[-1])
        uvicorn.run(self.app, host=host, port=port, log_level="error")

    def get_last_n_kb(self, input_string, n=20):
        num_chars = round(n * 1024)
        last_n_kb = input_string[-num_chars:]
        return last_n_kb

    def sendLog(self, level, message, log_name="logger"):

        if IsSetEvent("Error"):
            return

        log_obj = process_logger if log_name == "process_logger" else logger

        message = self.get_last_n_kb(str(message))
        getattr(log_obj, level.lower())(message)

        if not self.debugMode:
            log_obj.info(f"[sendLog] Send log level: {level} message: {message}")
            try:
                data = {"level": level, "message": message}
                response = requests.post(f"{OPERATOR_REST_API_URI}/LogMessage", json=data)
                assert (
                    response.status_code == 200
                ), f"Failed to post data. Status code: {response.status_code}, Error: {response.text}"
                log_obj.info(f"[sendLog] Log sending succeeds, response: {response.text}")
            except Exception as err:
                log_obj.error(f"[sendLog] Exception: {err}")

        if level == "ERROR":
            log_obj.info("[sendLog] Set edge alive event")

            SetEvent("Error")
            self.AliveEvent.set()

            if not self.debugMode:
                while True:
                    time.sleep(1)

    def subprocessLog(self, process):

        _, stderr = process.communicate()
        if stderr and not IsSetEvent("ProcessFinished"):
            self.sendLog("ERROR", f"[subprocessLog] Exception: {stderr}", "process_logger")

        CleanEvent("ProcessFinished")

    def close_process(self):

        if isinstance(self.mainProcess, subprocess.Popen):
            try:
                logger.info("[close_process] Close training process.")
                self.mainProcess.terminate()
                self.mainProcess.kill()
            except Exception as err:
                logger.info(f"[close_process] Got error for closing training process: {err}")

        if isinstance(self.monitorProcess, mp.context.Process):
            try:
                logger.info("[close_process] Close monitor process.")
                self.monitorProcess.terminate()
                self.monitorProcess.join()
                if compareVersion(python_version(), "3.7") >= 0:
                    self.monitorProcess.close()
            except Exception as err:
                logger.info(f"[close_process] Got error for closing monitor process: {err}")

        if isinstance(self.p, mp.context.Process):
            try:
                logger.info("[close_process] Close func process.")
                self.p.terminate()
                self.p.join()
                if compareVersion(python_version(), "3.7") >= 0:
                    self.p.close()
            except Exception as err:
                logger.info(f"[close_process] Got error for closing func process: {err}")


class EdgeApp(BaseAPP):
    def __init__(self, mainProcess, preProcess, debugMode=False):

        super().__init__(mainProcess=mainProcess, preProcess=preProcess, debugMode=debugMode)

        self.app.add_api_route("/DataValidate", self.data_validate, methods=["POST"])
        self.app.add_api_route("/TrainInit", self.train_init, methods=["POST"])
        self.app.add_api_route("/LocalTrain", self.local_train, methods=["POST"])
        self.app.add_api_route("/TrainInterrupt", self.train_interrupt, methods=["POST"])
        self.app.add_api_route("/TrainFinish", self.train_finish, methods=["POST"])

        self.schema = self.__load_schema(os.getenv("SCHEMA_PATH"))

    def data_validate(self, request: Request):

        logger.info("[DataValidate] Start DataValidate.")

        try:
            CleanAllEvent()
            CleanInfoJson()
            if os.path.exists(os.environ["LOCAL_MODEL_PATH"]):
                os.remove(os.environ["LOCAL_MODEL_PATH"])
        except Exception as err:
            self.sendLog("ERROR", f"[DataValidate] Exception: {err}")

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    def train_init(self, request: Request):

        logger.info("[TrainInit] Start TrainInit.")

        if isinstance(self.mainProcess, subprocess.Popen):
            logger.warning("[TrainInit] Process already exists.")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "Training process already exists."},
            )

        if self.preProcess:
            process = None
            try:
                logger.info("[TrainInit] Start data preprocessing.")
                logger.info("[TrainInit] {}.".format(self.preProcess))

                process = subprocess.Popen(
                    [ele for ele in self.preProcess.split(" ") if ele.strip()],
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
                process.wait()

                _, stderr = process.communicate()
                if stderr and not IsSetEvent("ProcessFinished"):
                    self.sendLog("ERROR", f"[TrainInit] CalledProcessError: {stderr}")
                CleanEvent("ProcessFinished")

                if IsSetEvent("Error"):
                    return JSONResponse(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        content={"message": f"{stderr}"},
                    )

            except Exception as err:
                self.sendLog("ERROR", f"[TrainInit] Exception: {err}")

        try:
            logger.info("[TrainInit] Start training process.")
            logger.info("[TrainInit] {}.".format(self.mainProcess))

            self.mainProcess = subprocess.Popen(
                [ele for ele in self.mainProcess.split(" ") if ele.strip()],
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

        except Exception as err:
            self.sendLog("ERROR", f"[TrainInit] Exception: {err}")

        try:
            logger.info("[TrainInit] Start monitor process.")
            self.monitorProcess = mp.Process(
                target=self.subprocessLog, kwargs={"process": self.mainProcess}
            )
            self.monitorProcess.start()
        except Exception as err:
            self.sendLog("ERROR", f"[TrainInit] Exception: {err}")

        logger.info("[TrainInit] Wait for TrainInitDone.")
        WaitEvent("TrainInitDone")
        logger.info("[TrainInit] Get TrainInitDone.")

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    def local_train(self, request: LocalTrainRequest):

        logger.info("[LocalTrain] Start LocalTrain.")

        if self.mainProcess.poll() is not None:
            self.sendLog("ERROR", "[LocalTrain] Training process has been terminated.")

        if not self.debugMode and isinstance(self.p, mp.context.Process):
            logger.info("[LocalTrain] Close previous func process.")
            try:
                self.p.terminate()
                self.p.join()
                if compareVersion(python_version(), "3.7") >= 0:
                    self.p.close()
            except Exception:
                pass

        self.p = mp.Process(target=self.__func_local_train)
        self.p.start()
        if self.debugMode:
            self.p.join()

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    def train_interrupt(self, request: Request):
        # Not Implemented
        logger.info("[TrainInterrupt] Start TrainInterrupt.")
        self.AliveEvent.set()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    def train_finish(self, request: Request):
        logger.info("[TrainFinish] Start TrainFinish.")
        self.AliveEvent.set()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    def __func_local_train(self):

        logger.info("[LocalTrain] Trainer has been called to start training.")
        SetEvent("TrainStarted")

        logger.info("[LocalTrain] Wait until the training has done.")
        WaitEvent("TrainFinished")

        try:
            logger.info("[LocalTrain] Read info from training process.")
            with open(
                os.path.join(os.path.dirname(os.environ["LOCAL_MODEL_PATH"]), "info.json"), "r"
            ) as openfile:
                info = json.load(openfile)
        except Exception as err:
            self.sendLog("ERROR", f"[LocalTrain] Exception: {err}")

        try:
            validate(instance=info, schema=self.schema)
            logger.info(
                "[LocalTrain] model datasetSize: {}".format(info["metadata"]["datasetSize"])
            )
            logger.info("[LocalTrain] model metrics: {}".format(info["metrics"]))
        except Exception:
            self.sendLog("ERROR", "[LocalTrain] Exception: Json Schema Error")

        if not self.debugMode:

            try:
                response = requests.post(f"{OPERATOR_REST_API_URI}/LocalTrainFinish", json=info)
                assert (
                    response.status_code == 200
                ), f"Failed to post data. Status code: {response.status_code}, Error: {response.text}"
                logger.info("[LocalTrain] Response sent.")
            except Exception as err:
                self.sendLog("ERROR", f"[LocalTrain] Error: {err}")

        logger.info("[LocalTrain] Trainer finish training.")

    def __load_schema(self, path):
        with open(path, "r") as openfile:
            return json.load(openfile)


class AggregatorApp(BaseAPP):
    def __init__(self, mainProcess, debugMode=False, init_once=False):

        super().__init__(mainProcess=mainProcess, debugMode=debugMode)

        self.init_once = init_once
        self.repo_root = os.environ.get("REPO_ROOT", "/repos")
        self.global_model_path = None
        self.global_info_path = None

    def aggregate(self, request: AggregateRequest):

        self.sendLog("INFO", "[aggregate] Start Aggregate.")

        if not self.debugMode and isinstance(self.p, mp.context.Process):
            logger.info("[aggregate] Close previous func process.")
            try:
                self.p.terminate()
                self.p.join()
                if compareVersion(python_version(), "3.7") >= 0:
                    self.p.close()
            except Exception:
                pass

        try:
            self.__prepare_resources(request)

        except Exception as err:
            self.sendLog("ERROR", f"[aggregate] Exception: {err}")

        if self.init_once:

            if not isinstance(self.mainProcess, subprocess.Popen):
                try:
                    logger.info("[aggregate] Start aggregator process.")
                    logger.info("[aggregate] {}.".format(self.mainProcess))

                    self.mainProcess = subprocess.Popen(
                        [ele for ele in self.mainProcess.split(" ") if ele.strip()],
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                    )

                except Exception as err:
                    self.sendLog("ERROR", f"[aggregate] Exception: {err}")

                try:
                    logger.info("[aggregate] Start monitor process.")
                    self.monitorProcess = mp.Process(
                        target=self.subprocessLog, kwargs={"process": self.mainProcess}
                    )
                    self.monitorProcess.start()
                except Exception as err:
                    self.sendLog("ERROR", f"[aggregate] Exception: {err}")

            if self.mainProcess.poll() is not None:
                self.sendLog("ERROR", "[aggregate] Aggregator process has been terminated.")

        self.p = mp.Process(target=self.__func_aggregate)
        self.p.start()
        if self.debugMode:
            self.p.join()

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    def train_finish(self, request: Request):
        logger.info("[train_finish] Start TrainFinish.")
        self.AliveEvent.set()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    def __func_aggregate(self):

        if not self.init_once:

            process = None
            try:
                logger.info("[aggregate] Start aggregator process.")
                logger.info("[aggregate] {}.".format(self.mainProcess))

                process = subprocess.Popen(
                    [ele for ele in self.mainProcess.split(" ") if ele.strip()],
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
                process.wait()
                _, stderr = process.communicate()
                if stderr and not IsSetEvent("ProcessFinished"):
                    self.sendLog("ERROR", f"[Aggregate] CalledProcessError: {stderr}")
                CleanEvent("ProcessFinished")

                logger.info("[Aggregate] Aggregator process finish.")
            except Exception as err:
                self.sendLog("ERROR", f"[Aggregate] Exception: {err}")

        else:
            logger.info("[aggregate] Aggregator has been called to start aggregating.")
            SetEvent("AggregateStarted")

            logger.info("[aggregate] Wait until the aggregation has done.")
            WaitEvent("AggregateFinished")
            logger.info("[aggregate] Get AggregateFinished.")

        if not self.debugMode:
            try:
                logger.info("[aggregate] Read info from training process.")
                with open(self.global_info_path, "r") as openfile:
                    global_info = json.load(openfile)
                response = requests.post(
                    f"{OPERATOR_REST_API_URI}/AggregateFinish", json=global_info
                )
                assert (
                    response.status_code == 200
                ), f"Failed to post data. Status code: {response.status_code}, Error: {response.text}"
                logger.info("[aggregate] Response sent.")
            except Exception as err:
                self.sendLog("ERROR", f"[Aggregate] Error: {err}")

        self.sendLog("INFO", "[aggregate] Aggregation succeeds.")

    def __prepare_resources(self, request):

        logger.info("[aggregate] Check local model checkpoint")
        models = []
        for m in request.LocalModels:
            path = os.path.join(self.repo_root, m.path, "weights.ckpt")
            if not os.path.isfile(path):
                raise FileNotFoundError(path + " not found")
            models.append(path)
        SetPaths("localModels", models)
        logger.info("[aggregate] Check local info json")
        infos = []
        for m in request.LocalModels:
            path = os.path.join(self.repo_root, m.path, "info.json")
            if not os.path.isfile(path):
                raise FileNotFoundError(path + " not found")
            infos.append(path)
        SetPaths("localInfos", infos)
        with open(infos[0], "r") as openfile:
            infoData = json.load(openfile)
        epoch = infoData["metadata"]["epoch"]

        logger.info("[aggregate] Set globalModel")
        self.global_model_path = os.path.join(
            self.repo_root, request.AggregatedModel.path, "merged.ckpt"
        )
        SetPaths("globalModel", self.global_model_path)

        logger.info("[aggregate] Set globalInfo")
        self.global_info_path = os.path.join(
            self.repo_root, request.AggregatedModel.path, "merged-info.json"
        )
        SetPaths("globalInfo", self.global_info_path)
        SaveGlobalInfoJson(infos, self.global_info_path)

        self.sendLog("INFO", f"[Aggregate] Before aggregation. epoch: {epoch}")


def run_app(app):

    server_process = mp.Process(target=app.run)
    server_process.start()

    app.AliveEvent.wait()
    app.close_process()

    time.sleep(10)

    server_process.terminate()
    server_process.join()
    if compareVersion(python_version(), "3.7") >= 0:
        server_process.close()

    os._exit(os.EX_OK)
