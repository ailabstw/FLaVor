import json
import logging
import os
import subprocess
from concurrent import futures
from multiprocessing import Event, Process

import grpc

from . import service_pb2, service_pb2_grpc
from .utils import SetEvent, WaitEvent

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"

log_format = logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
logger_path = "/log/appService.log"
os.makedirs(os.path.dirname(logger_path), exist_ok=True)
handler = logging.FileHandler(logger_path, mode="w", encoding=None, delay=False)
handler.setFormatter(log_format)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

process_logger = logging.getLogger("process")
process_logger.setLevel(logging.INFO)
process_logger_path = "/log/appProcess.log"
os.makedirs(os.path.dirname(process_logger_path), exist_ok=True)
process_handler = logging.FileHandler(process_logger_path, mode="w", encoding=None, delay=False)
process_handler.setFormatter(log_format)
process_handler.setLevel(logging.INFO)
process_logger.addHandler(process_handler)


class EdgeAppServicer(service_pb2_grpc.EdgeAppServicer):
    def __init__(self):

        self.__OPERATOR_URI = os.getenv("OPERATOR_URI") or "127.0.0.1:8787"
        self.__channel = grpc.insecure_channel(self.__OPERATOR_URI)
        logger.info(f"[init] grpc.insecure_channel: {self.__OPERATOR_URI} Done.")
        self.__stub = service_pb2_grpc.EdgeOperatorStub(self.__channel)
        logger.info("[init] service_pb2_grpc.EdgeOperatorStub Done.")

        self.EdgeAliveEvent = Event()

        self.dataPreProcess = None
        self.trainingProcess = None

    def IsDataValidated(self, request, context):
        resp = service_pb2.Empty()
        logger.info(f"[IsDataValidated] Sending response: {resp}")
        return resp

    def TrainInit(self, request, context):

        if isinstance(self.trainingProcess, subprocess.Popen):
            resp = service_pb2.Empty()
            logger.info("[TrainInit] Process already exists.")
            logger.info(f"[TrainInit] Sending response: {resp}")
            return resp

        if self.dataPreProcess:
            logger.info("[TrainInit] Start data preprocessing.")
            logger.info("[TrainInit] {}.".format(self.dataPreProcess))
            try:
                subprocess.check_output(
                    [ele for ele in self.dataPreProcess.split(" ") if ele.strip()],
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError as e:
                err = e.output.decode("utf-8")
                logger.error(f"[TrainInit] CalledProcessError: {err}")
                self.__sendLog("ERROR", err)
                WaitEvent("Error")

        logger.info("[TrainInit] Start training process.")
        logger.info("[TrainInit] {}.".format(self.trainingProcess))
        self.trainingProcess = subprocess.Popen(
            [ele for ele in self.trainingProcess.split(" ") if ele.strip()],
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        logger.info("[TrainInit] Start monitor process.")
        try:
            self.monitorProcess = Process(
                target=self.__subprocessLog, kwargs={"process": self.trainingProcess}
            )
            self.monitorProcess.start()
        except Exception as err:
            logger.error(f"[TrainInit] Exception: {err}")
            self.__sendLog("ERROR", str(err))

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
            self.__sendLog("ERROR", str(err))
            WaitEvent("Error")

        logger.info("[LocalTrain] model datasetSize: {}".format(info["metadata"]["datasetSize"]))
        logger.info("[LocalTrain] model metrics: {}".format(info["metrics"]))

        try:
            result = service_pb2.LocalTrainResult(
                error=0, metadata=info["metadata"], metrics=info["metrics"]
            )
            response = self.__stub.LocalTrainFinish(result, timeout=30)  # noqa F841
        except grpc.RpcError as rpc_error:
            logger.error(f"[LocalTrain] RpcError: {rpc_error}")
            self.__sendLog("ERROR", str(rpc_error))
            WaitEvent("Error")
        except Exception as err:
            logger.error(f"[LocalTrain] Exception: {err}")
            self.__sendLog("ERROR", str(err))
            WaitEvent("Error")

        resp = service_pb2.Empty()
        logger.info(f"[LocalTrain] Sending response: {resp}")
        return resp

    def TrainInterrupt(self, request, context):
        # Not Implemented
        logger.info("[TrainInterrupt] TrainInterrupt")
        self.EdgeAliveEvent.set()
        return service_pb2.Empty()

    def TrainFinish(self, _request, _context):
        logger.info("[TrainFinish] TrainFinish")
        self.EdgeAliveEvent.set()
        return service_pb2.Empty()

    def __sendLog(self, level, message):

        logger.info(f"[__sendLog] Send grpc log level: {level} message: {message}")
        try:
            message = service_pb2.Log(level=level, message=message)
            response = self.__stub.LogMessage(message)  # noqa F841
            logger.info(f"[__sendLog] Log sending succeeds, response: {response}")
        except grpc.RpcError as rpc_error:
            logger.error(f"[__sendLog] RpcError: {rpc_error}")
        except Exception as err:
            logger.error(f"[__sendLog] Exception: {err}")

        if level == "ERROR":
            logger.info("[__sendLog] Set edge alive event")
            self.EdgeAliveEvent.set()

    def __sendprocessLog(self, level, message):

        process_logger.info(f"[__sendprocessLog] Send grpc log level: {level} message: {message}")
        channel = grpc.insecure_channel(self.__OPERATOR_URI)
        process_logger.info("[__sendprocessLog] grpc.insecure_channel for multiprocess Done.")
        stub = service_pb2_grpc.EdgeOperatorStub(channel)
        process_logger.info(
            "[__sendprocessLog] service_pb2_grpc.EdgeOperatorStub for multiprocess Done."
        )
        try:
            message = service_pb2.Log(level=level, message=message)
            response = stub.LogMessage(message)  # noqa F841
            process_logger.info(f"[__sendprocessLog] Log sending succeeds, response: {response}")
        except grpc.RpcError as rpc_error:
            process_logger.error(f"[__sendprocessLog] RpcError: {rpc_error}")
        except Exception as err:
            process_logger.error(f"[__sendprocessLog] Exception: {err}")

        process_logger.info("[__sendprocessLog] Close multiprocess grpc channel.")
        channel.close()

        if level == "ERROR":
            process_logger.info("[__sendprocessLog] Set edge alive event")
            self.EdgeAliveEvent.set()

    def __subprocessLog(self, process):

        _, stderr = process.communicate()
        if stderr:
            process_logger.error(f"[__subprocessLog] Exception: {stderr}")
            self.__sendprocessLog("ERROR", str(stderr))

    def close_service(self):

        logger.info("[close_service] Close training process.")
        try:
            self.trainingProcess.terminate()
            self.trainingProcess.kill()
        except Exception as err:
            logger.error(f"[close_service] Got error for closing training preprocess: {err}")

        logger.info("[close_service] Close monitor process.")
        try:
            self.monitorProcess.terminate()
            self.monitorProcess.join()
            self.monitorProcess.close()
        except Exception as err:
            logger.error(f"[close_service] Got error for closing monitor preprocess: {err}")

        logger.info("[close_service] Close grpc channel.")
        self.__channel.close()

        logger.info("[close_service] Close service done.")


def serve(app_service):

    APPLICATION_URI = "0.0.0.0:7878"

    logger.info(f"[serve] Start server... {APPLICATION_URI}")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_EdgeAppServicer_to_server(app_service, server)
    server.add_insecure_port(APPLICATION_URI)
    server.start()

    app_service.EdgeAliveEvent.wait()
    app_service.close_service()

    server.stop(None)
    os._exit(os.EX_OK)
