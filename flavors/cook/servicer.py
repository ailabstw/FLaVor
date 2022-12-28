import logging
import multiprocessing as mp
import os
import shutil
from concurrent import futures
from multiprocessing import Process

import grpc

from . import service_pb2, service_pb2_grpc
from .log_msg import LogLevel, PackLogMsg, UnPackLogMsg

log_filename = "/log/appService.log"
file_handler = logging.FileHandler(log_filename, mode="w", encoding=None, delay=False)
logging.basicConfig(
    filename=log_filename,
    format="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

class EdgeAppServicer(service_pb2_grpc.EdgeAppServicer):
    def __init__(self):

        self.__OPERATOR_URI = os.getenv("OPERATOR_URI") or "127.0.0.1:8787"
        self.__channel = grpc.insecure_channel(self.__OPERATOR_URI)
        logging.info(f"grpc.insecure_channel: {self.__OPERATOR_URI} Done.")
        self.__stub = service_pb2_grpc.EdgeOperatorStub(self.__channel)
        logging.info("service_pb2_grpc.EdgeOperatorStub Done.")

        self.trainInitDoneEvent = mp.Event()
        self.trainStartedEvent = mp.Event()
        self.trainFinishedEvent = mp.Event()
        self.namespace = mp.Manager().Namespace()
        self.logQueue = mp.Queue()

        self.EdgeAliveEvent = mp.Event()

        self.loggingProcess = Process(target=self.__logEventLoop)
        self.loggingProcess.start()
        self.dataPreProcess = None
        self.trainingProcess = None

    def IsDataValidated(self, request, context):
        resp = service_pb2.Empty()
        logging.info(f"Sending response: {resp}")
        return resp

    def TrainInit(self, request, context):
        logging.info("TrainInit")

        try:
            if self.dataPreProcess:
                logging.info("Run data preprocess.")
                self.dataPreProcess.start()
                self.dataPreProcess.join()
                self.dataPreProcess.close()

            self.namespace.localModelPath = os.environ["LOCAL_MODEL_PATH"]
            self.namespace.globalModelPath = os.environ["GLOBAL_MODEL_PATH"]

            logging.info("Run training preprocess.")
            self.trainingProcess.start()

            self.trainInitDoneEvent.wait()

        except Exception as err:
            self.logQueue.put(PackLogMsg(LogLevel.ERROR, str(err)))

        resp = service_pb2.Empty()
        logging.info(f"Sending response: {resp}")
        return resp

    def LocalTrain(self, request, context):
        logging.info("LocalTrain")

        try:
            logging.info(f"pretrained (global) model path: [{self.namespace.globalModelPath}]")
            logging.info(f"local model path: [{self.namespace.localModelPath}]")
            logging.info(f"epoch count: [{request.EpR}]")

            logging.info("trainer has been called to start training.")
            self.trainStartedEvent.set()

            logging.info("wait until the training has done.")
            self.trainFinishedEvent.wait()
            logging.info("training finished event clear.")
            self.trainFinishedEvent.clear()

            logging.info(f"model last epoch path: [{self.namespace.epoch_path}]")
            shutil.copyfile(self.namespace.epoch_path, self.namespace.localModelPath)

            logging.info("model datasetSize: {}".format(self.namespace.metadata["datasetSize"]))
            logging.info(f"model metrics: {self.namespace.metrics}")
            logging.info(f"config.GRPC_CLIENT_URI: {self.__OPERATOR_URI}")

            result = service_pb2.LocalTrainResult(
                error=0, metadata=self.namespace.metadata, metrics=self.namespace.metrics
            )
            logging.info("service_pb2.LocalTrainResult Done.")
            response = self.__stub.LocalTrainFinish(result, timeout=30)
            logging.info("stub.LocalTrainFinish Done.")
            logging.debug(f"sending grpc message succeeds, response: {response}")

        except Exception as err:
            self.logQueue.put(PackLogMsg(LogLevel.ERROR, str(err)))

        resp = service_pb2.Empty()
        logging.info(f"Sending response: {resp}")
        return resp

    def TrainInterrupt(self, request, context):
        # Not Implemented
        logging.info("TrainInterrupt")
        self.EdgeAliveEvent.set()
        return service_pb2.Empty()

    def TrainFinish(self, _request, _context):
        logging.info("TrainFinish")
        self.EdgeAliveEvent.set()
        return service_pb2.Empty()

    def __logEventLoop(self):

        while True:
            obj = self.logQueue.get()
            level, message = UnPackLogMsg(obj)
            logging.info(f"Send log level: {level} message: {message}")
            message = service_pb2.Log(level=level, message=message)
            try:
                response = self.__stub.LogMessage(message)
                logging.info(f"Log sending succeeds, response: {response}")
            except grpc.RpcError as rpc_error:
                logging.error(f"grpc error: {rpc_error}")
            except Exception as err:
                logging.error(f"got error: {err}")
            if level == "ERROR":
                self.EdgeAliveEvent.set()
                return

    def close_service(self):

        if self.dataPreProcess and not self.dataPreProcess._closed:
            try:
                self.dataPreProcess.terminate()
                self.dataPreProcess.join()
                self.dataPreProcess.close()
                logging.info("close data preprocess.")
            except Exception as err:
                logging.error(f"got error for closing data preprocess: {err}")

        try:
            self.trainingProcess.terminate()
            self.trainingProcess.join()
            self.trainingProcess.close()
            logging.info("close training process.")
        except Exception as err:
            logging.error(f"got error for closing training preprocess: {err}")

        try:
            self.loggingProcess.terminate()
            self.loggingProcess.join()
            self.loggingProcess.close()
            logging.info("close logging process.")
        except Exception as err:
            logging.error(f"got error for closing logging preprocess: {err}")

        self.__channel.close()
        logging.info("channel.close() Done.")


def serve(app_service):

    APPLICATION_URI = "0.0.0.0:7878"

    logging.info(f"Start server... {APPLICATION_URI}")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_EdgeAppServicer_to_server(app_service, server)
    server.add_insecure_port(APPLICATION_URI)
    server.start()

    app_service.EdgeAliveEvent.wait()
    app_service.close_service()

    server.stop(None)
    os._exit(os.EX_OK)
