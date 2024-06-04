import json
import logging
import multiprocessing as mp
import os
import subprocess
import threading
import time
from platform import python_version

import requests
import uvicorn
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from .model import AggregateRequest, FLResponse, LocalTrainRequest
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
        self.monitorThread = None

        self.AliveEvent = mp.Event()
        self.is_error = mp.Value("i", 0)

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

        log_obj = process_logger if log_name == "process_logger" else logger

        message = self.get_last_n_kb(str(message))
        getattr(log_obj, level.lower())(message)

        if not self.debugMode:
            log_obj.info(f"[sendLog] Send log level: {level} message: {message}")
            try:
                data = {"level": level, "message": message}
                response = requests.post(f"http://{OPERATOR_REST_API_URI}/LogMessage", json=data)
                assert (
                    response.status_code == 200
                ), f"Failed to post data. Status code: {response.status_code}, Error: {response.text}"
                log_obj.info(f"[sendLog] Log sending succeeds, response: {response.text}")
            except Exception as err:
                log_obj.error(f"[sendLog] Exception: {err}")

        if level == "ERROR":

            self.is_error.value = 1
            self.close_process()

            log_obj.info("[sendLog] Set edge alive event")
            self.AliveEvent.set()

    def subprocessLog(self):

        _, stderr = self.mainProcess.communicate()
        if stderr and not IsSetEvent("ProcessFinished"):
            self.sendLog("ERROR", f"[subprocessLog] Exception: {stderr}", "process_logger")

        CleanEvent("ProcessFinished")

    def close_process(self):

        if isinstance(self.mainProcess, subprocess.Popen):
            try:
                logger.info("[close_process] Close training process.")
                self.mainProcess.terminate()
                self.mainProcess.kill()
            except Exception:
                pass


class EdgeApp(BaseAPP):
    def __init__(self, mainProcess, preProcess, debugMode=False):

        super().__init__(mainProcess=mainProcess, preProcess=preProcess, debugMode=debugMode)

        self.app.add_api_route("/DataValidate", self.data_validate, methods=["POST"])
        self.app.add_api_route("/TrainInit", self.train_init, methods=["POST"])
        self.app.add_api_route("/LocalTrain", self.local_train, methods=["POST"])
        self.app.add_api_route("/TrainInterrupt", self.train_interrupt, methods=["POST"])
        self.app.add_api_route("/TrainFinish", self.train_finish, methods=["POST"])

    def data_validate(self, request: Request):

        logger.info("[data_validate] Start DataValidate.")

        try:
            CleanAllEvent()
            CleanInfoJson()
            if os.path.exists(os.environ["LOCAL_MODEL_PATH"]):
                os.remove(os.environ["LOCAL_MODEL_PATH"])
        except Exception as err:
            self.sendLog("ERROR", f"[DataValidate] Exception: {err}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": f"{err}"},
            )

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    def train_init(self, request: Request):

        logger.info("[train_init] Start TrainInit.")

        if isinstance(self.mainProcess, subprocess.Popen):
            logger.warning("[train_init] Process already exists.")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "Training process already exists."},
            )

        if self.preProcess:
            process = None
            try:
                logger.info("[train_init] Start data preprocessing.")
                logger.info("[train_init] {}.".format(self.preProcess))

                process = subprocess.Popen(
                    [ele for ele in self.preProcess.split(" ") if ele.strip()],
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
                process.wait()

                _, stderr = process.communicate()
                if stderr and not IsSetEvent("ProcessFinished"):
                    self.sendLog("ERROR", f"[TrainInit] CalledProcessError: {stderr}")
                    return JSONResponse(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        content={"message": f"{stderr}"},
                    )
                CleanEvent("ProcessFinished")

            except Exception as err:
                self.sendLog("ERROR", f"[TrainInit] Exception: {err}")
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"message": f"{err}"},
                )

        try:
            logger.info("[train_init] Start training process.")
            logger.info("[train_init] {}.".format(self.mainProcess))

            self.mainProcess = subprocess.Popen(
                [ele for ele in self.mainProcess.split(" ") if ele.strip()],
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

        except Exception as err:
            self.sendLog("ERROR", f"[TrainInit] Exception: {err}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": f"{err}"},
            )

        try:
            logger.info("[train_init] Start monitor thread.")
            self.monitorThread = threading.Thread(target=self.subprocessLog)
            self.monitorThread.start()
        except Exception as err:
            self.sendLog("ERROR", f"[TrainInit] Exception: {err}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": f"{err}"},
            )

        logger.info("[train_init] Wait for TrainInitDone.")
        WaitEvent("TrainInitDone", self.is_error)
        if self.is_error.value != 0:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": "Error in TrainInit Stage."},
            )
        logger.info("[train_init] Get TrainInitDone.")

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    def local_train(self, request: LocalTrainRequest):

        logger.info("[local_train] Start LocalTrain.")

        try:
            if self.mainProcess.poll() is not None:
                self.sendLog("ERROR", "[local_train] Training process has been terminated.")
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"message": "Training process has been terminated"},
                )
        except Exception as err:
            self.sendLog("ERROR", f"[local_train] Exception: {err}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": f"{err}"},
            )

        t = threading.Thread(target=self.__func_local_train)
        t.start()
        if self.debugMode:
            t.join()

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    def train_interrupt(self, request: Request):
        # Not Implemented
        logger.info("[train_interrupt] Start TrainInterrupt.")
        self.close_process()
        self.AliveEvent.set()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    def train_finish(self, request: Request):
        logger.info("[train_finish] Start TrainFinish.")
        self.close_process()
        self.AliveEvent.set()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    def __func_local_train(self):

        logger.info("[local_train] Trainer has been called to start training.")
        SetEvent("TrainStarted")

        logger.info("[local_train] Wait until the training has done.")
        WaitEvent("TrainFinished", self.is_error)
        if self.is_error.value != 0:
            return

        try:
            logger.info("[local_train] Read info from training process.")
            with open(
                os.path.join(os.path.dirname(os.environ["LOCAL_MODEL_PATH"]), "info.json"), "r"
            ) as openfile:
                info = json.load(openfile)

            FLResponse.model_validate(info)
            logger.info(
                "[local_train] model datasetSize: {}".format(info["metadata"]["datasetSize"])
            )
            logger.info("[local_train] model metrics: {}".format(info["metrics"]))

            if not self.debugMode:
                response = requests.post(
                    f"http://{OPERATOR_REST_API_URI}/LocalTrainFinish", json=info
                )
                assert (
                    response.status_code == 200
                ), f"Failed to post data. Status code: {response.status_code}, Error: {response.text}"
                logger.info("[local_train] Response sent.")

            logger.info("[local_train] Trainer finish training.")

        except Exception as err:
            self.sendLog("ERROR", f"[LocalTrain] Exception: {err}")


class AggregatorApp(BaseAPP):
    def __init__(self, mainProcess, debugMode=False, init_once=False):

        super().__init__(mainProcess=mainProcess, debugMode=debugMode)

        self.app.add_api_route("/Aggregate", self.aggregate, methods=["POST"])
        self.app.add_api_route("/TrainFinish", self.train_finish, methods=["POST"])

        self.init_once = init_once
        self.repo_root = os.environ.get("REPO_ROOT", "/repos")
        self.global_model_path = None
        self.global_info_path = None

    def aggregate(self, request: AggregateRequest):

        self.sendLog("INFO", "[aggregate] Start Aggregate.")

        try:
            self.__prepare_resources(request)

        except Exception as err:
            self.sendLog("ERROR", f"[aggregate] Exception: {err}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": f"{err}"},
            )

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
                    return JSONResponse(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        content={"message": f"{err}"},
                    )

                try:
                    logger.info("[aggregate] Start monitor thread.")
                    self.monitorThread = threading.Thread(target=self.subprocessLog)
                    self.monitorThread.start()
                except Exception as err:
                    self.sendLog("ERROR", f"[aggregate] Exception: {err}")
                    return JSONResponse(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        content={"message": f"{err}"},
                    )

            if self.mainProcess.poll() is not None:
                self.sendLog("ERROR", "[aggregate] Aggregator process has been terminated.")
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"message": "Aggregator process has been terminated"},
                )

        t = threading.Thread(target=self.__func_aggregate)
        t.start()
        if self.debugMode:
            t.join()

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    def train_finish(self, request: Request):
        logger.info("[train_finish] Start TrainFinish.")
        self.close_process()
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
                    return
                CleanEvent("ProcessFinished")

                logger.info("[Aggregate] Aggregator process finish.")
            except Exception as err:
                self.sendLog("ERROR", f"[Aggregate] Exception: {err}")
                return

        else:
            logger.info("[aggregate] Aggregator has been called to start aggregating.")
            SetEvent("AggregateStarted")

            logger.info("[aggregate] Wait until the aggregation has done.")
            WaitEvent("AggregateFinished", self.is_error)
            if self.is_error.value != 0:
                return
            logger.info("[aggregate] Get AggregateFinished.")

        try:
            logger.info("[aggregate] Read info from training process.")
            with open(self.global_info_path, "r") as openfile:
                global_info = json.load(openfile)
            FLResponse.model_validate(global_info)

            if not self.debugMode:
                response = requests.post(
                    f"http://{OPERATOR_REST_API_URI}/AggregateFinish", json=global_info
                )
                assert (
                    response.status_code == 200
                ), f"Failed to post data. Status code: {response.status_code}, Error: {response.text}"
                logger.info("[aggregate] Response sent.")

        except Exception as err:
            self.sendLog("ERROR", f"[Aggregate] Error: {err}")
            return

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

    time.sleep(10)

    server_process.terminate()
    server_process.join()
    if compareVersion(python_version(), "3.7") >= 0:
        server_process.close()

    os._exit(os.EX_OK)
