from . import batcher


class TaiMedimgProcessor:
    def __init__(self, callback):

        self.ready = False

        self.batcher_server = batcher.BatcherServer(callback, 1)
        self.batcher_server.start()

        self.ready = True

    def __call__(self, **kwargs):

        try:
            response_data = self.infer_handler(**kwargs)
        except Exception as e:
            response_data = self.error_handler(str(e))

        return response_data

    def error_handler(self, error):
        return {"error": error}

    def infer_handler(self, **kwargs):
        infer_result = self.batcher_server.create_client()._predict_and_wait([kwargs])[0]
        if isinstance(infer_result, Exception):
            raise infer_result
        return infer_result
