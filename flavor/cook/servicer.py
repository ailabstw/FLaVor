import json
import logging
import multiprocessing as mp
import os
import subprocess
import time
from concurrent import futures
from platform import python_version

import grpc
from jsonschema import validate

from . import service_pb2, service_pb2_grpc
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

        self.GRPC_RETRY_NUM = 5
        self.GRPC_RETRY_INTERVAL = 5

        self.preProcess = preProcess
        self.mainProcess = mainProcess
        self.monitorProcess = None
        self.p = None

        self.AliveEvent = mp.Event()

        self.debugMode = debugMode

    def get_last_n_kb(self, input_string, n=20):
        num_chars = round(n * 1024)
        last_n_kb = input_string[-num_chars:]
        return last_n_kb

    def sendLog(self, level, message, log_name="logger"):

        if IsSetEvent("Error"):
            return

        log_obj = eval(log_name)

        message = self.get_last_n_kb(str(message))
        getattr(log_obj, level.lower())(message)

        if not self.debugMode:
            log_obj.info(f"[sendLog] Send grpc log level: {level} message: {message}")

            channel = grpc.insecure_channel(OPERATOR_URI)
            stub = None
            if self.__class__.__name__ == "EdgeAppServicer":
                stub = service_pb2_grpc.EdgeOperatorStub(channel)
            elif self.__class__.__name__ == "AggregateServerAppServicer":
                stub = service_pb2_grpc.AggregateServerOperatorStub(channel)
            else:
                log_obj.error(f"[sendLog] Servicer {self.__class__.__name__ } not supported.")

            cnt = 0
            while cnt < self.GRPC_RETRY_NUM:
                try:
                    message_pkg = service_pb2.Log(level=level, message=message)
                    response = stub.LogMessage(message_pkg)  # noqa F841
                    log_obj.info(f"[sendLog] Log sending succeeds, response: {response}")
                    break
                except grpc.RpcError as rpc_error:
                    log_obj.error(f"[sendLog] RpcError: {rpc_error}")
                except Exception as err:
                    log_obj.error(f"[sendLog] Exception: {err}")
                time.sleep(self.GRPC_RETRY_INTERVAL)
                cnt += 1
                log_obj.info(f"[sendLog] GRPC retry {cnt}")
            channel.close()

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

    def close_service(self):

        if isinstance(self.mainProcess, subprocess.Popen):
            try:
                logger.info("[close_service] Close training process.")
                self.mainProcess.terminate()
                self.mainProcess.kill()
            except Exception as err:
                logger.info(f"[close_service] Got error for closing training process: {err}")

        if isinstance(self.monitorProcess, mp.context.Process):
            try:
                logger.info("[close_service] Close monitor process.")
                self.monitorProcess.terminate()
                self.monitorProcess.join()
                if compareVersion(python_version(), "3.7") >= 0:
                    self.monitorProcess.close()
            except Exception as err:
                logger.info(f"[close_service] Got error for closing monitor process: {err}")

        if isinstance(self.p, mp.context.Process):
            try:
                logger.info("[close_service] Close func process.")
                self.p.terminate()
                self.p.join()
                if compareVersion(python_version(), "3.7") >= 0:
                    self.p.close()
            except Exception as err:
                logger.info(f"[close_service] Got error for closing func process: {err}")

        logger.info("[close_service] Close service done.")


class EdgeAppServicer(BaseServicer, service_pb2_grpc.EdgeAppServicer):
    def __init__(self, mainProcess, preProcess, debugMode=False):

        super().__init__(mainProcess=mainProcess, preProcess=preProcess, debugMode=debugMode)

        with open(os.environ["SCHEMA_PATH"], "r") as openfile:
            self.schema = json.load(openfile)

    def DataValidate(self, request, context):

        logger.info("[DataValidate] Start DataValidate.")

        try:
            CleanAllEvent()
            CleanInfoJson()
            if os.path.exists(os.environ["LOCAL_MODEL_PATH"]):
                os.remove(os.environ["LOCAL_MODEL_PATH"])
        except Exception as err:
            self.sendLog("ERROR", f"[DataValidate] Exception: {err}")

        return service_pb2.Empty()

    def TrainInit(self, request, context):

        logger.info("[TrainInit] Start TrainInit.")

        if isinstance(self.mainProcess, subprocess.Popen):
            logger.warning("[TrainInit] Process already exists.")
            return service_pb2.Empty()

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
            except Exception as err:
                self.sendLog("ERROR", f"[TrainInit] Exception: {err}")

            _, stderr = process.communicate()
            if stderr and not IsSetEvent("ProcessFinished"):
                self.sendLog("ERROR", f"[TrainInit] CalledProcessError: {stderr}")
            CleanEvent("ProcessFinished")

            if IsSetEvent("Error"):
                return service_pb2.Empty()

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

        return service_pb2.Empty()

    def LocalTrain(self, request, context):

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

        return service_pb2.Empty()

    def TrainInterrupt(self, request, context):
        # Not Implemented
        logger.info("[TrainInterrupt] Start TrainInterrupt.")
        self.AliveEvent.set()
        return service_pb2.Empty()

    def TrainFinish(self, _request, _context):
        logger.info("[TrainFinish] Start TrainFinish.")
        self.AliveEvent.set()
        return service_pb2.Empty()

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

            channel = None
            stub = None
            rpc_msg, except_msg = None, None

            try:
                channel = grpc.insecure_channel(OPERATOR_URI)
                stub = service_pb2_grpc.EdgeOperatorStub(channel)
            except Exception as err:
                self.sendLog("ERROR", f"[LocalTrain] Exception: {err}")

            cnt = 0
            while cnt < self.GRPC_RETRY_NUM:
                try:
                    result = service_pb2.LocalTrainResult(
                        error=0, metadata=info["metadata"], metrics=info["metrics"]
                    )
                    response = stub.LocalTrainFinish(result, timeout=30)  # noqa F841
                    logger.info("[LocalTrain] Response sent.")
                    break
                except grpc.RpcError as rpc_error:
                    if cnt == self.GRPC_RETRY_NUM - 1:
                        rpc_msg = rpc_error
                except Exception as err:
                    except_msg = err
                    break

                cnt += 1
                if cnt < self.GRPC_RETRY_NUM:
                    time.sleep(self.GRPC_RETRY_INTERVAL)
                    logger.info(f"[LocalTrain] GRPC retry {cnt}")

            channel.close()

            if rpc_msg is not None:
                self.sendLog("ERROR", f"[LocalTrain] RpcError: {rpc_msg}")
            elif except_msg is not None:
                self.sendLog("ERROR", f"[LocalTrain] RpcError: {except_msg}")
            rpc_msg, except_msg = None, None
        logger.info("[LocalTrain] Trainer finish training.")


class AggregateServerAppServicer(BaseServicer, service_pb2_grpc.AggregateServerAppServicer):
    def __init__(self, mainProcess, debugMode=False, init_once=False):

        super().__init__(mainProcess=mainProcess, debugMode=debugMode)

        self.init_once = init_once
        self.repo_root = os.environ.get("REPO_ROOT", "/repos")

    def Aggregate(self, request, context):

        self.sendLog("INFO", "[Aggregate] Start Aggregate.")

        if not self.debugMode and isinstance(self.p, mp.context.Process):
            logger.info("[Aggregate] Close previous func process.")
            try:
                self.p.terminate()
                self.p.join()
                if compareVersion(python_version(), "3.7") >= 0:
                    self.p.close()
            except Exception:
                pass

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
            self.sendLog("ERROR", f"[Aggregate] Exception: {err}")

        self.sendLog("INFO", f"[Aggregate] Before aggregation. epoch: {epoch}")

        if self.init_once:

            if not isinstance(self.mainProcess, subprocess.Popen):
                try:
                    logger.info("[Aggregate] Start aggregator process.")
                    logger.info("[Aggregate] {}.".format(self.mainProcess))
                    self.mainProcess = subprocess.Popen(
                        [ele for ele in self.mainProcess.split(" ") if ele.strip()],
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                    )
                except Exception as err:
                    self.sendLog("ERROR", f"[Aggregate] Exception: {err}")

                try:
                    logger.info("[Aggregate] Start monitor process.")
                    self.monitorProcess = mp.Process(
                        target=self.subprocessLog, kwargs={"process": self.mainProcess}
                    )
                    self.monitorProcess.start()
                except Exception as err:
                    self.sendLog("ERROR", f"[Aggregate] Exception: {err}")

            if self.mainProcess.poll() is not None:
                self.sendLog("ERROR", "[Aggregate] Aggregator process has been terminated.")

        self.p = mp.Process(target=self.__func_aggregate)
        self.p.start()
        if self.debugMode:
            self.p.join()

        return service_pb2.Empty()

    def TrainFinish(self, request, context):
        logger.info("[TrainFinish] Start TrainFinish.")
        self.AliveEvent.set()
        return service_pb2.Empty()

    def __func_aggregate(self):

        if not self.init_once:

            process = None
            try:
                logger.info("[Aggregate] Start aggregator process.")
                logger.info("[Aggregate] {}.".format(self.mainProcess))
                process = subprocess.Popen(
                    [ele for ele in self.mainProcess.split(" ") if ele.strip()],
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
                process.wait()
                logger.info("[Aggregate] Aggregator process finish.")
            except Exception as err:
                self.sendLog("ERROR", f"[Aggregate] Exception: {err}")

            _, stderr = process.communicate()
            if stderr and not IsSetEvent("ProcessFinished"):
                self.sendLog("ERROR", f"[Aggregate] CalledProcessError: {stderr}")
            CleanEvent("ProcessFinished")
        else:
            logger.info("[Aggregate] Aggregator has been called to start aggregating.")
            SetEvent("AggregateStarted")

            logger.info("[Aggregate] Wait until the aggregation has done.")
            WaitEvent("AggregateFinished")
            logger.info("[Aggregate] Get AggregateFinished.")

        if not self.debugMode:

            channel = None
            stub = None
            rpc_msg, except_msg = None, None

            try:
                channel = grpc.insecure_channel(OPERATOR_URI)
                stub = service_pb2_grpc.AggregateServerOperatorStub(channel)
            except Exception as err:
                self.sendLog("ERROR", f"[Aggregate] Exception: {err}")

            cnt = 0
            while cnt < self.GRPC_RETRY_NUM:
                try:
                    result = service_pb2.AggregateResult(error=0)
                    response = stub.AggregateFinish(result)  # noqa F841
                    logger.info("[Aggregate] Response sent.")
                    break
                except grpc.RpcError as rpc_error:
                    if cnt == self.GRPC_RETRY_NUM - 1:
                        rpc_msg = rpc_error
                except Exception as err:
                    except_msg = err
                    break
                cnt += 1
                if cnt < self.GRPC_RETRY_NUM:
                    time.sleep(self.GRPC_RETRY_INTERVAL)
                    logger.info(f"[Aggregate] GRPC retry {cnt}")
            channel.close()

            if rpc_msg is not None:
                self.sendLog("ERROR", f"[Aggregate] RpcError: {rpc_msg}")
            elif except_msg is not None:
                self.sendLog("ERROR", f"[Aggregate] Exception: {except_msg}")
            rpc_msg, except_msg = None, None

        self.sendLog("INFO", "[Aggregate] Aggregation succeeds.")


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
        servicer.close_service()
        time.sleep(10)

        server.stop(None)

    os._exit(os.EX_OK)
