# -*- coding: utf-8 -*-
import logging
import threading
from collections import deque

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _create_buffer():
    return {"result": deque(), "trigger": threading.Event()}


class BatcherServer:
    def __init__(self, inferer, batch_size):

        self.inferer = inferer
        self.batch_size = batch_size

        # client info for server
        self.result_buffer_dict = (
            dict()
        )  # to push result into client's result buffer by key: client_id

        # client push local_batch into input_buffer and trigger input_trigger for server to process
        self.input_buffer = deque()
        # deque of tuple : (data, client_id, trigger_or_not)
        # trigger_or_not : last data in local batch set trigger_or_not = TRUE
        #                  => push into trigger_back => trigger client to get the 'entire' result for the local batch
        self.input_trigger = threading.Event()

        # for server to popleft and detuple from input_buffer and get a batch to feed into model
        self.vacancy = self.batch_size
        self.batch = []  # gether the data batch to feed into model
        self.owner_id = []  # keep client_id for each data to push the result to its result buffer
        self.trigger_back = (
            set()
        )  # keep the client_id to trigger it to get its 'entire' result for the local batch

        self.prediction_thread = threading.Thread(name="prediction_thread", target=self.__run)
        self.run_flag = False

    def __del__(self):  # you have to make sure no client is using
        self.stop()

    def get_batch_size(self):
        return self.batch_size

    def create_client(self):
        return BatcherClient(self)

    # public
    def start(self):
        self.run_flag = True
        self.prediction_thread.setDaemon(True)
        self.prediction_thread.start()

    def stop(self):
        # digest the left
        while self.input_buffer:  # not empty
            pass
        self.run_flag = False
        self.input_trigger.set()

    # friend function to client
    def _join(self, client_id):
        logging.info("[Batcher] [Server] client {} wait join".format(client_id))
        result_buffer_ref = _create_buffer()
        self.result_buffer_dict[client_id] = result_buffer_ref
        logging.info("[Batcher] [Server] client {} join".format(client_id))

        return result_buffer_ref

    def _quit(self, client_id):
        self.result_buffer_dict.pop(client_id, None)
        logging.info("[Batcher] [Server] client {} quit".format(client_id))

    def _push_batch(self, local_batch):
        [self.input_buffer.append(rec) for rec in local_batch]
        self.input_trigger.set()

    def __run(self):
        while self.run_flag:
            self.input_trigger.wait()
            self.input_trigger.clear()
            self.__batch_process()

    def _batch_predict(self, batch):
        result = []
        for batch_item in batch:
            try:
                result.append(self.inferer(**batch_item))
            except Exception as e:
                result.append(e)
        return result

    def __batch_process(self):
        logging.info("[Batcher] [Server] has client {}".format(self.result_buffer_dict.keys()))
        while self.input_buffer:  # not empty
            self.vacancy = self.batch_size
            self.batch = []
            self.owner_id = []
            self.trigger_back = set()

            while self.input_buffer and self.vacancy > 0:
                # use pop rather than iterate to ensure input_buffer concurrency
                (data, client_id, trigger_flag) = self.input_buffer.popleft()
                self.batch.append(data)
                self.owner_id.append(client_id)
                if trigger_flag:
                    self.trigger_back.add(client_id)
                self.vacancy -= 1
            logging.info("[Batcher] [Server] process client {}".format(self.owner_id))
            logging.info("[Batcher] [Server] trigger client {}".format(self.trigger_back))
            batch_result = self._batch_predict(self.batch)

            # push result back to client's result_buffer
            for oid, res in zip(self.owner_id, batch_result):
                try:  # need check key exist when push result
                    self.result_buffer_dict[oid]["result"].append(res)
                    logging.info("[Batcher] [Server] client {} push result".format(oid))
                except Exception:
                    logging.error(
                        "[Batcher] [Server] [BAD] client {} push result failed".format(oid)
                    )
                    pass

            # trigger the client to get 'entire' result for the local_batch
            for key in self.trigger_back:
                try:  # need check key exist when push result
                    self.result_buffer_dict[key]["trigger"].set()
                    logging.info("[Batcher] [Server] client {} trigger".format(key))
                except Exception:
                    logging.error("[Batcher] [Server] [BAD] client {} trigger failed".format(key))
                    pass


class BatcherClient:
    def __init__(self, server):
        self.server = server
        self.client_id = id(self)
        self.buffer = self.server._join(self.client_id)
        self.result_buffer = self.buffer["result"]
        logging.info("[Batcher] [Client {}] __init__".format(self.client_id))

    def __del__(self):
        self.server._quit(self.client_id)
        logging.info("[Batcher] [Client {}] __del__".format(self.client_id))

    def get_batch_size(self):
        return self.server.get_batch_size()

    def _predict_and_wait(self, input_list):
        local_batch = [(input, self.client_id, False) for input in input_list[:-1]]
        local_batch.append((input_list[-1], self.client_id, True))  # True with the last input
        self.server._push_batch(local_batch)
        logging.info("[Batcher] [Client {}] wait".format(self.client_id))
        self.buffer["trigger"].wait()
        self.buffer["trigger"].clear()
        logging.info("[Batcher] [Client {}] get result".format(self.client_id))
        result_list = []
        while self.result_buffer:
            result_list.append(self.result_buffer.popleft())

        return result_list
