import json
import logging
import os
import subprocess
import time
from concurrent import futures
from multiprocessing import Event, Process
from platform import python_version

import grpc
from jsonschema import validate

from . import service_pb2, service_pb2_grpc
from .utils import (
    CleanAllEvent,
    CleanInfoJson,
    SaveGlobalInfoJson,
    SetEvent,
    SetPaths,
    WaitEvent,
    compareVersion,
)

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"

APPLICATION_URI = os.getenv("APPLICATION_URI", "0.0.0.0:7878")
OPERATOR_URI = os.getenv("OPERATOR_URI", "127.0.0.1:8787")

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


class BaseServicer:
    def __init__(self, mainProcess, preProcess=None, debugMode=False):

        super().__init__()

        self.stub = None
        self.channel = None
        self.GRPC_RETRY_NUM = 5
        self.GRPC_RETRY_INTERVAL = 5

        self.preProcess = preProcess
        self.mainProcess = mainProcess
        self.monitorProcess = None

        self.AliveEvent = Event()

        self.debugMode = debugMode

    def sendLog(self, level, message):

        message = str(message)

        if not self.debugMode:
            logger.info(f"[sendLog] Send grpc log level: {level} message: {message}")

            cnt = 0
            while cnt < self.GRPC_RETRY_NUM:
                try:
                    message_pkg = service_pb2.Log(level=level, message=message)
                    response = self.stub.LogMessage(message_pkg)  # noqa F841
                    logger.info(f"[sendLog] Log sending succeeds, response: {response}")
                    break
                except grpc.RpcError as rpc_error:
                    logger.error(f"[sendLog] RpcError: {rpc_error}")
                except Exception as err:
                    logger.error(f"[sendLog] Exception: {err}")
                time.sleep(self.GRPC_RETRY_INTERVAL)
                cnt += 1
                logger.info(f"[sendLog] GRPC retry {cnt}")

        if level == "ERROR":
            logger.info("[sendLog] Set edge alive event")
            self.close_service()
            if self.debugMode:
                SetEvent("Error")
            else:
                WaitEvent("Error")

    def sendprocessLog(self, level, message):

        message = str(message)

        if not self.debugMode:
            process_logger.info(f"[sendprocessLog] Send grpc log level: {level} message: {message}")
            channel = grpc.insecure_channel(OPERATOR_URI)
            process_logger.info("[sendprocessLog] grpc.insecure_channel for multiprocess Done.")
            stub = getattr(service_pb2_grpc, self.stub.__class__.__name__)(channel)
            process_logger.info(
                f"[sendprocessLog] service_pb2_grpc.{self.stub.__class__.__name__} for multiprocess Done."
            )

            cnt = 0
            while cnt < self.GRPC_RETRY_NUM:
                try:
                    message_pkg = service_pb2.Log(level=level, message=message)
                    response = stub.LogMessage(message_pkg)  # noqa F841
                    process_logger.info(
                        f"[sendprocessLog] Log sending succeeds, response: {response}"
                    )
                    break
                except grpc.RpcError as rpc_error:
                    process_logger.error(f"[sendprocessLog] RpcError: {rpc_error}")
                except Exception as err:
                    process_logger.error(f"[sendprocessLog] Exception: {err}")
                time.sleep(self.GRPC_RETRY_INTERVAL)
                cnt += 1
                logger.info(f"[sendprocessLog] GRPC retry {cnt}")

            process_logger.info("[sendprocessLog] Close multiprocess grpc channel.")
            channel.close()

        if level == "ERROR":
            process_logger.info("[sendprocessLog] Set edge alive event")
            self.close_service()
            if self.debugMode:
                SetEvent("Error")
            else:
                WaitEvent("Error")

    def subprocessLog(self, process):

        _, stderr = process.communicate()
        if stderr:
            process_logger.error(f"[subprocessLog] Exception: {stderr}")
            self.sendprocessLog("ERROR", stderr)

    def close_service(self):

        logger.info("[close_service] Close training process.")
        try:
            self.mainProcess.terminate()
            self.mainProcess.kill()
        except Exception as err:
            logger.info(f"[close_service] Got error for closing training preprocess: {err}")

        logger.info("[close_service] Close monitor process.")

        if self.monitorProcess:
            try:
                self.monitorProcess.terminate()
                self.monitorProcess.join()
                if compareVersion(python_version(), "3.7") >= 0:
                    self.monitorProcess.close()
            except Exception as err:
                logger.info(f"[close_service] Got error for closing monitor preprocess: {err}")

        logger.info("[close_service] Close grpc channel.")
        self.channel.close()

        logger.info("[close_service] Close service done.")
        self.AliveEvent.set()


class EdgeAppServicer(BaseServicer, service_pb2_grpc.EdgeAppServicer):
    def __init__(self, mainProcess, preProcess, debugMode=False):

        super().__init__(mainProcess=mainProcess, preProcess=preProcess, debugMode=debugMode)

        self.channel = grpc.insecure_channel(OPERATOR_URI)
        logger.info(f"[init] grpc.insecure_channel: {OPERATOR_URI} Done.")
        self.stub = service_pb2_grpc.EdgeOperatorStub(self.channel)
        logger.info("[init] service_pb2_grpc.EdgeOperatorStub Done.")

        with open(os.environ["SCHEMA_PATH"], "r") as openfile:
            self.schema = json.load(openfile)

    def DataValidate(self, request, context):

        CleanAllEvent()
        CleanInfoJson()
        if os.path.exists(os.environ["LOCAL_MODEL_PATH"]):
            os.remove(os.environ["LOCAL_MODEL_PATH"])

        resp = service_pb2.Empty()
        logger.info(f"[IsDataValidated] Sending response: {resp}")
        return resp

    def TrainInit(self, request, context):

        if isinstance(self.mainProcess, subprocess.Popen):
            resp = service_pb2.Empty()
            logger.warning("[TrainInit] Process already exists.")
            logger.info(f"[TrainInit] Sending response: {resp}")
            return resp

        if self.preProcess:
            logger.info("[TrainInit] Start data preprocessing.")
            logger.info("[TrainInit] {}.".format(self.preProcess))
            process = subprocess.Popen(
                [ele for ele in self.preProcess.split(" ") if ele.strip()],
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            process.wait()
            _, stderr = process.communicate()
            if stderr:
                logger.error(f"[TrainInit] CalledProcessError: {stderr}")
                self.sendLog("ERROR", stderr)

        logger.info("[TrainInit] Start training process.")
        logger.info("[TrainInit] {}.".format(self.mainProcess))
        try:
            self.mainProcess = subprocess.Popen(
                [ele for ele in self.mainProcess.split(" ") if ele.strip()],
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
        except Exception as err:
            logger.error(f"[TrainInit] Exception: {err}")
            self.sendLog("ERROR", err)

        logger.info("[TrainInit] Start monitor process.")
        try:
            self.monitorProcess = Process(
                target=self.subprocessLog, kwargs={"process": self.mainProcess}
            )
            self.monitorProcess.start()
        except Exception as err:
            logger.error(f"[TrainInit] Exception: {err}")
            self.sendLog("ERROR", err)

        logger.info("[TrainInit] Wait for TrainInitDone.")
        WaitEvent("TrainInitDone")

        resp = service_pb2.Empty()
        logger.info(f"[TrainInit] Sending response: {resp}")
        return resp

    def LocalTrain(self, request, context):

        if self.mainProcess.poll() is not None:
            logger.error("[LocalTrain] Training process has been terminated.")
            self.sendLog("ERROR", "Training process has been terminated")

        logger.info("[LocalTrain] Trainer has been called to start training.")
        SetEvent("TrainStarted")

        logger.info("[LocalTrain] Wait until the training has done.")
        WaitEvent("TrainFinished")

        logger.info("[LocalTrain] Read info from training process.")
        try:
            with open(
                os.path.join(os.path.dirname(os.environ["LOCAL_MODEL_PATH"]), "info.json"), "r"
            ) as openfile:
                info = json.load(openfile)
        except Exception as err:
            logger.error(f"[LocalTrain] Exception: {err}")
            self.sendLog("ERROR", err)

        try:
            validate(instance=info, schema=self.schema)
        except Exception:
            logger.error("[LocalTrain] Exception: Json Schema Error")
            self.sendLog("ERROR", "Json Schema Error")

        logger.info("[LocalTrain] model datasetSize: {}".format(info["metadata"]["datasetSize"]))
        logger.info("[LocalTrain] model metrics: {}".format(info["metrics"]))

        if not self.debugMode:
            cnt = 0
            while cnt < self.GRPC_RETRY_NUM:
                try:
                    result = service_pb2.LocalTrainResult(
                        error=0, metadata=info["metadata"], metrics=info["metrics"]
                    )
                    response = self.stub.LocalTrainFinish(result, timeout=30)  # noqa F841
                    break
                except grpc.RpcError as rpc_error:
                    logger.error(f"[LocalTrain] RpcError: {rpc_error}")
                    if cnt == self.GRPC_RETRY_NUM - 1:
                        self.sendLog("ERROR", rpc_error)
                except Exception as err:
                    logger.error(f"[LocalTrain] Exception: {err}")
                    self.sendLog("ERROR", err)
                    break
                time.sleep(self.GRPC_RETRY_INTERVAL)
                cnt += 1
                logger.info(f"[LocalTrain] GRPC retry {cnt}")

        resp = service_pb2.Empty()
        logger.info(f"[LocalTrain] Sending response: {resp}")
        return resp

    def TrainInterrupt(self, request, context):
        # Not Implemented
        logger.info("[TrainInterrupt] TrainInterrupt")
        self.close_service()
        return service_pb2.Empty()

    def TrainFinish(self, _request, _context):
        logger.info("[TrainFinish] TrainFinish")
        self.close_service()
        return service_pb2.Empty()


class AggregateServerAppServicer(BaseServicer, service_pb2_grpc.AggregateServerAppServicer):
    def __init__(self, mainProcess, debugMode=False, init_once=False):

        super().__init__(mainProcess=mainProcess, debugMode=debugMode)

        self.init_once = init_once
        self.channel = grpc.insecure_channel(OPERATOR_URI)
        logger.info(f"[init] grpc.insecure_channel: {OPERATOR_URI} Done.")
        self.stub = service_pb2_grpc.AggregateServerOperatorStub(self.channel)
        logger.info("[init] service_pb2_grpc.AggregateServerOperatorStub Done.")

        self.repo_root = os.environ.get("REPO_ROOT", "/repos")

    def Aggregate(self, request, context):

        logger.info("[Aggregate] Aggregate start")
        self.sendLog("INFO", "Received aggregation result")

        try:
            logger.info("[Aggregate] Check local model checkpoint")
            models = []
            for m in request.localModels:
                path = os.path.join(self.repo_root, m.path, "weights.ckpt")
                if not os.path.isfile(path):
                    raise FileNotFoundError(path + " not found")
                models.append(path)
            SetPaths("localModels", models)

            logger.info("[Aggregate] Check local info json")
            infos = []
            for m in request.localModels:
                path = os.path.join(self.repo_root, m.path, "info.json")
                if not os.path.isfile(path):
                    raise FileNotFoundError(path + " not found")
                infos.append(path)
            SetPaths("localInfos", infos)

            with open(infos[0], "r") as openfile:
                infoData = json.load(openfile)
            epoch = infoData["metadata"]["epoch"]

            logger.info("[Aggregate] Set globalModel")
            SetPaths(
                "globalModel",
                os.path.join(self.repo_root, request.aggregatedModel.path, "merged.ckpt"),
            )

            logger.info("[Aggregate] Set globalInfo")
            global_info_path = os.path.join(
                self.repo_root, request.aggregatedModel.path, "merged-info.json"
            )
            SetPaths("globalInfo", global_info_path)

            SaveGlobalInfoJson(infos, global_info_path)

        except Exception as err:
            logging.error(f"[Aggregate] Exception: {err}")
            self.sendLog("ERROR", f"AggregationError: {err}")

        logger.info(f"[Aggregate] Before aggregation. epoch: {epoch}")
        self.sendLog("INFO", f"Before aggregation. epoch: {epoch}")

        if not self.init_once:
            logger.info("[Aggregate] Start aggregator process.")
            logger.info("[Aggregate] {}.".format(self.mainProcess))

            process = subprocess.Popen(
                [ele for ele in self.mainProcess.split(" ") if ele.strip()],
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            process.wait()
            _, stderr = process.communicate()
            if stderr:
                logger.error(f"[Aggregate] CalledProcessError: {stderr}")
                self.sendLog("ERROR", stderr)

        elif not isinstance(self.mainProcess, subprocess.Popen):
            logger.info("[Aggregate] Start aggregator process.")
            logger.info("[Aggregate] {}.".format(self.mainProcess))
            try:
                self.mainProcess = subprocess.Popen(
                    [ele for ele in self.mainProcess.split(" ") if ele.strip()],
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
            except Exception as err:
                logger.error(f"[Aggregate] Exception: {err}")
                self.sendLog("ERROR", err)

            logger.info("[Aggregate] Start monitor process.")
            try:
                self.monitorProcess = Process(
                    target=self.subprocessLog, kwargs={"process": self.mainProcess}
                )
                self.monitorProcess.start()
            except Exception as err:
                logger.error(f"[Aggregate] Exception: {err}")
                self.sendLog("ERROR", err)

        if self.init_once:

            if self.mainProcess.poll() is not None:
                logger.error("[Aggregate] Aggregator process has been terminated.")
                self.sendLog("ERROR", "Aggregator process has been terminated")

            logger.info("[Aggregate] Aggregator has been called to start aggregating.")
            SetEvent("AggregateStarted")

            logger.info("[Aggregate] Wait until the aggregation has done.")
            WaitEvent("AggregateFinished")

        logger.info(f"[Aggregate] Aggregation succeeds. epoch {epoch}")
        self.sendLog("INFO", f"Aggregation succeeds. epoch {epoch}")

        if not self.debugMode:
            cnt = 0
            while cnt < self.GRPC_RETRY_NUM:
                try:
                    result = service_pb2.AggregateResult(error=0)
                    response = self.stub.AggregateFinish(result)  # noqa F841
                    break
                except grpc.RpcError as rpc_error:
                    logger.error(f"[Aggregate] RpcError: {rpc_error}")
                    if cnt == self.GRPC_RETRY_NUM - 1:
                        self.sendLog("ERROR", rpc_error)
                except Exception as err:
                    logger.error(f"[Aggregate] Exception: {err}")
                    self.sendLog("ERROR", err)
                    break
                time.sleep(self.GRPC_RETRY_INTERVAL)
                cnt += 1
                logger.info(f"[Aggregate] GRPC retry {cnt}")

        resp = service_pb2.Empty()
        logging.info(f"[AggregateServerServicer] Aggregate sending response: {resp}")
        return resp

    def TrainFinish(self, request, context):
        logging.info("[AggregateServerServicer] received TrainFinish message")
        self.close_service()
        return service_pb2.Empty()


def serve(servicer, stype):

    PROTO_VERSION = os.getenv("PROTO_VERSION")
    EXPECT_PROTO_VERSION = "1.0"

    if PROTO_VERSION != EXPECT_PROTO_VERSION:
        logger.error(
            f"[checkProtoVersion] Expect proto version: {EXPECT_PROTO_VERSION}, got {PROTO_VERSION}"
        )
    else:
        if stype not in ["client", "server"]:
            logger.error(f"[serve] Unsupport type of servicer: {stype}")
            raise ValueError(f"Unsupport type of servicer: {stype}")

        logger.info(f"[serve] Start {stype} server... {APPLICATION_URI}")

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        if stype == "client":
            service_pb2_grpc.add_EdgeAppServicer_to_server(servicer, server)
        else:
            service_pb2_grpc.add_AggregateServerAppServicer_to_server(servicer, server)

        server.add_insecure_port(APPLICATION_URI)
        server.start()

        servicer.AliveEvent.wait()
        time.sleep(10)

        server.stop(None)

    os._exit(os.EX_OK)
