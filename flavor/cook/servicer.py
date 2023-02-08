import json
import logging
import os
import subprocess
from concurrent import futures
from multiprocessing import Event, Process
from platform import python_version

import grpc

from . import service_pb2, service_pb2_grpc
from .utils import SaveGlobalInfoJson, SetEvent, SetPaths, WaitEvent, compareVersion

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"

log_format = logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
logger_path = os.path.join(os.environ["LOG_PATH"], "appService.log")
handler = logging.FileHandler(logger_path, mode="w", encoding=None, delay=False)
handler.setFormatter(log_format)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

process_logger = logging.getLogger("process")
process_logger.setLevel(logging.INFO)
process_logger_path = os.path.join(os.environ["LOG_PATH"], "appProcess.log")
process_handler = logging.FileHandler(process_logger_path, mode="w", encoding=None, delay=False)
process_handler.setFormatter(log_format)
process_handler.setLevel(logging.INFO)
process_logger.addHandler(process_handler)


class BaseServicer:
    def __init__(self):

        self.OPERATOR_URI = os.getenv("OPERATOR_URI", "127.0.0.1:8787")
        self.stub = None
        self.channel = None

        self.preProcess = None
        self.mainProcess = None
        self.monitorProcess = None

        self.AliveEvent = Event()

    def sendLog(self, level, message):

        logger.info(f"[sendLog] Send grpc log level: {level} message: {message}")
        try:
            message = service_pb2.Log(level=level, message=message)
            response = self.stub.LogMessage(message)  # noqa F841
            logger.info(f"[sendLog] Log sending succeeds, response: {response}")
        except grpc.RpcError as rpc_error:
            logger.error(f"[sendLog] RpcError: {rpc_error}")
        except Exception as err:
            logger.error(f"[sendLog] Exception: {err}")

        if level == "ERROR":
            logger.info("[sendLog] Set edge alive event")
            self.AliveEvent.set()

    def sendprocessLog(self, level, message):

        process_logger.info(f"[sendprocessLog] Send grpc log level: {level} message: {message}")
        channel = grpc.insecure_channel(self.OPERATOR_URI)
        process_logger.info("[sendprocessLog] grpc.insecure_channel for multiprocess Done.")
        stub = getattr(service_pb2_grpc, self.stub.__class__.__name__)(channel)
        process_logger.info(
            f"[sendprocessLog] service_pb2_grpc.{self.stub.__class__.__name__} for multiprocess Done."
        )
        try:
            message = service_pb2.Log(level=level, message=message)
            response = stub.LogMessage(message)  # noqa F841
            process_logger.info(f"[sendprocessLog] Log sending succeeds, response: {response}")
        except grpc.RpcError as rpc_error:
            process_logger.error(f"[sendprocessLog] RpcError: {rpc_error}")
        except Exception as err:
            process_logger.error(f"[sendprocessLog] Exception: {err}")

        process_logger.info("[sendprocessLog] Close multiprocess grpc channel.")
        channel.close()

        if level == "ERROR":
            process_logger.info("[sendprocessLog] Set edge alive event")
            self.AliveEvent.set()

    def subprocessLog(self, process):

        _, stderr = process.communicate()
        if stderr:
            process_logger.error(f"[subprocessLog] Exception: {stderr}")
            self.sendprocessLog("ERROR", str(stderr))

    def close_service(self):

        logger.info("[close_service] Close training process.")
        try:
            self.mainProcess.terminate()
            self.mainProcess.kill()
        except Exception as err:
            logger.error(f"[close_service] Got error for closing training preprocess: {err}")

        logger.info("[close_service] Close monitor process.")

        if self.monitorProcess:
            try:
                self.monitorProcess.terminate()
                self.monitorProcess.join()
                if compareVersion(python_version(), "3.7") >= 0:
                    self.monitorProcess.close()
            except Exception as err:
                logger.error(f"[close_service] Got error for closing monitor preprocess: {err}")

        logger.info("[close_service] Close grpc channel.")
        self.channel.close()

        logger.info("[close_service] Close service done.")


class EdgeAppServicer(BaseServicer, service_pb2_grpc.EdgeAppServicer):
    def __init__(self):

        super().__init__()

        self.channel = grpc.insecure_channel(self.OPERATOR_URI)
        logger.info(f"[init] grpc.insecure_channel: {self.OPERATOR_URI} Done.")
        self.stub = service_pb2_grpc.EdgeOperatorStub(self.channel)
        logger.info("[init] service_pb2_grpc.EdgeOperatorStub Done.")

    def DataValidate(self, request, context):
        resp = service_pb2.Empty()
        logger.info(f"[IsDataValidated] Sending response: {resp}")
        return resp

    def TrainInit(self, request, context):

        if isinstance(self.mainProcess, subprocess.Popen):
            resp = service_pb2.Empty()
            logger.info("[TrainInit] Process already exists.")
            logger.info(f"[TrainInit] Sending response: {resp}")
            return resp

        if self.preProcess:
            logger.info("[TrainInit] Start data preprocessing.")
            logger.info("[TrainInit] {}.".format(self.preProcess))
            try:
                subprocess.check_output(
                    [ele for ele in self.preProcess.split(" ") if ele.strip()],
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError as e:
                err = e.output.decode("utf-8")
                logger.error(f"[TrainInit] CalledProcessError: {err}")
                self.sendLog("ERROR", err)
                WaitEvent("Error")

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
            self.sendLog("ERROR", str(err))

        logger.info("[TrainInit] Start monitor process.")
        try:
            self.monitorProcess = Process(
                target=self.subprocessLog, kwargs={"process": self.mainProcess}
            )
            self.monitorProcess.start()
        except Exception as err:
            logger.error(f"[TrainInit] Exception: {err}")
            self.sendLog("ERROR", str(err))

        logger.info("[TrainInit] Wait for TrainInitDone.")
        WaitEvent("TrainInitDone")

        resp = service_pb2.Empty()
        logger.info(f"[TrainInit] Sending response: {resp}")
        return resp

    def LocalTrain(self, request, context):

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
            self.sendLog("ERROR", str(err))
            WaitEvent("Error")

        logger.info("[LocalTrain] model datasetSize: {}".format(info["metadata"]["datasetSize"]))
        logger.info("[LocalTrain] model metrics: {}".format(info["metrics"]))

        try:
            result = service_pb2.LocalTrainResult(
                error=0, metadata=info["metadata"], metrics=info["metrics"]
            )
            response = self.stub.LocalTrainFinish(result, timeout=30)  # noqa F841
        except grpc.RpcError as rpc_error:
            logger.error(f"[LocalTrain] RpcError: {rpc_error}")
            self.sendLog("ERROR", str(rpc_error))
            WaitEvent("Error")
        except Exception as err:
            logger.error(f"[LocalTrain] Exception: {err}")
            self.sendLog("ERROR", str(err))
            WaitEvent("Error")

        resp = service_pb2.Empty()
        logger.info(f"[LocalTrain] Sending response: {resp}")
        return resp

    def TrainInterrupt(self, request, context):
        # Not Implemented
        logger.info("[TrainInterrupt] TrainInterrupt")
        self.AliveEvent.set()
        return service_pb2.Empty()

    def TrainFinish(self, _request, _context):
        logger.info("[TrainFinish] TrainFinish")
        self.AliveEvent.set()
        return service_pb2.Empty()


class AggregateServerAppServicer(BaseServicer, service_pb2_grpc.AggregateServerAppServicer):
    def __init__(self, init_once=False):

        super().__init__()

        self.init_once = init_once
        self.channel = grpc.insecure_channel(self.OPERATOR_URI)
        logger.info(f"[init] grpc.insecure_channel: {self.OPERATOR_URI} Done.")
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
            WaitEvent("Error")

        logger.info(f"[Aggregate] Before aggregation. epoch: {epoch}")
        self.sendLog("INFO", f"Before aggregation. epoch: {epoch}")

        if not self.init_once:
            logger.info("[Aggregate] Start aggregator process.")
            logger.info("[Aggregate] {}.".format(self.mainProcess))
            try:
                subprocess.check_output(
                    [ele for ele in self.mainProcess.split(" ") if ele.strip()],
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError as e:
                err = e.output.decode("utf-8")
                logger.error(f"[Aggregate] CalledProcessError: {err}")
                self.sendLog("ERROR", err)
                WaitEvent("Error")

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
                self.sendLog("ERROR", str(err))
                WaitEvent("Error")

            logger.info("[Aggregate] Start monitor process.")
            try:
                self.monitorProcess = Process(
                    target=self.subprocessLog, kwargs={"process": self.mainProcess}
                )
                self.monitorProcess.start()
            except Exception as err:
                logger.error(f"[Aggregate] Exception: {err}")
                self.sendLog("ERROR", str(err))
                WaitEvent("Error")

        if self.init_once:
            logger.info("[Aggregate] Aggregator has been called to start aggregating.")
            SetEvent("AggregateStarted")

            logger.info("[Aggregate] Wait until the aggregation has done.")
            WaitEvent("AggregateFinished")

        logger.info(f"[Aggregate] Aggregation succeeds. epoch {epoch}")
        self.sendLog("INFO", f"Aggregation succeeds. epoch {epoch}")

        try:
            result = service_pb2.AggregateResult(error=0)
            response = self.stub.AggregateFinish(result)  # noqa F841
        except grpc.RpcError as rpc_error:
            logger.error(f"[Aggregate] RpcError: {rpc_error}")
            self.sendLog("ERROR", str(rpc_error))
            WaitEvent("Error")
        except Exception as err:
            logger.error(f"[Aggregate] Exception: {err}")
            self.sendLog("ERROR", str(err))
            WaitEvent("Error")

        resp = service_pb2.Empty()
        logging.info(f"[AggregateServerServicer] Aggregate sending response: {resp}")
        return resp

    def TrainFinish(self, request, context):
        logging.info("[AggregateServerServicer] received TrainFinish message")
        self.AliveEvent.set()
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

        APPLICATION_URI = os.getenv("APPLICATION_URI", "0.0.0.0:7878")
        logger.info(f"[serve] Start server... {APPLICATION_URI}")

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        if stype == "client":
            service_pb2_grpc.add_EdgeAppServicer_to_server(servicer, server)
        else:
            service_pb2_grpc.add_AggregateServerAppServicer_to_server(servicer, server)

        server.add_insecure_port(APPLICATION_URI)
        server.start()

        servicer.AliveEvent.wait()

        servicer.close_service()

        server.stop(None)

    os._exit(os.EX_OK)
