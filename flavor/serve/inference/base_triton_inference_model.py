import ctypes
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import tritonclient
import tritonclient.http
import tritonclient.utils.shared_memory as shm


class BaseTritonClient:
    """
    BaseTritonClient is a base class that sets up connections with Triton Inference Server.
    """

    DTYPES_CONFIG_TO_API = {
        "TYPE_BOOL": "BOOL",
        "TYPE_UINT8": "UINT8",
        "TYPE_UINT16": "UINT16",
        "TYPE_UINT32": "UINT32",
        "TYPE_UINT64": "UINT64",
        "TYPE_INT8": "INT8",
        "TYPE_INT16": "INT16",
        "TYPE_INT32": "INT32",
        "TYPE_INT64": "INT64",
        "TYPE_FP16": "FP16",
        "TYPE_FP32": "FP32",
        "TYPE_FP64": "FP64",
        "TYPE_STRING": "BYTES",
    }

    DTYPES_API_TO_NUMPY = {
      "BOOL": bool,
      "UINT8": np.uint8,
      "UINT16": np.uint16,
      "UINT32": np.uint32,
      "UINT64": np.uint64,
      "INT8": np.int8,
      "INT16": np.int16,
      "INT32": np.int32,
      "INT64": np.int64,
      "FP16": np.float16,
      "FP32": np.float32,
      "FP64": np.float64,
      "BYTES": np.object_,
    }

    def __init__(self, triton_url: str):
        self.triton_url = triton_url

        self.client = self._init_triton_client(triton_url)
        self.model_configs = self._load_model_configs()

    def _init_triton_client(self, triton_url: str) -> tritonclient.http.InferenceServerClient:
        """
        Make connection with Triton Inference Server.
        """
        client = tritonclient.http.InferenceServerClient(triton_url)
        try:
            client.is_server_ready()
            return client
        except Exception:
            raise ConnectionError(f"cannot connect to triton inference server at {triton_url}")

    def _load_model_configs(self) -> Dict[str, Any]:
        """
        Read current model configurations from Triton Inference Server.
        """
        models_status = self.client.get_model_repository_index()
        models = {}
        for model in models_status:
            name = model["name"]
            state = model["state"]

            if state != "READY":
                continue

            config = self.client.get_model_config(name)

            models[name] = {**model, **config}
        return models

    def get_model_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Read all model states.
        """
        self.model_configs = self._load_model_configs()
        states = {}
        for model in self.model_configs.values():
            inputs = [
                {
                    "name": inp["name"],
                    "data_type": inp["data_type"],
                    "dims": inp["dims"],
                }
                for inp in model["input"]
            ]
            outputs = [
                {
                    "name": out["name"],
                    "data_type": out["data_type"],
                    "dims": out["dims"],
                }
                for out in model["output"]
            ]

            states[model["name"]] = dict(
                name=model["name"],
                version=model["version"],
                state=model["state"],
                backend=model["backend"],
                max_batch_size=model["max_batch_size"],
                devices=[d["kind"] for d in model["instance_group"]],
                inputs=inputs,
                outputs=outputs,
            )
        return states

    def get_model_state(self, model_name: str) -> Dict[str, Any]:
        """
        Read model state of given model.
        """
        states = self.get_model_states()
        return states.get(model_name, {})


class TritonInferenceModel(BaseTritonClient):
    """
    TritonInferenceModel is a class that handles request inputs and response outputs.
    """

    def __init__(
        self,
        triton_url: str,
        model_name: str,
        model_version=str,
    ):
        super().__init__(triton_url)
        self.model_name = model_name
        self.model_version = model_version

        self.input_structs = None
        self.output_structs = None
        self.refresh_model_state()

    @property
    def model_state(self) -> Dict[str, Any]:
        return self.get_model_state(self.model_name)

    def refresh_model_state(self):
        self.input_structs = self.model_state["inputs"]
        self.output_structs = self.model_state["outputs"]

    def set_input_data(
        self, data_dict: Dict[str, np.ndarray]
    ) -> List[tritonclient.http.InferInput]:
        """
        turns input (`data_dict`) into triton input format

        Arguments
            data_dict: Dict[str, np.ndarray]
                key is input name, value is its content

        Returns
            List[tritonclient.http.InferInput]
                a list of triton inputs. order is not important.
        """
        inputs = []
        for input_struct in self.input_structs:
            name = input_struct["name"]
            data_type = input_struct["data_type"]
            data_type = BaseTritonClient.DTYPES_CONFIG_TO_API[data_type]
            dims = input_struct["dims"]
            if name in data_dict:
                if not isinstance(data_dict[name], np.ndarray):
                    data_dict[name] = np.asarray(data_dict[name])
                item: np.ndarray = data_dict[name]
                dims = item.shape
            else:
                raise KeyError(f"cannot find expected key {name}.")

            item = item.astype(self.DTYPES_API_TO_NUMPY[data_type])
            buffer = tritonclient.http.InferInput(name, dims, data_type)
            buffer.set_data_from_numpy(item)

            inputs.append(buffer)

        for key in data_dict.keys():
            if key not in {s["name"] for s in self.input_structs}:
                logging.warning(
                    f"found unwanted network input key={key}. make sure you know what you are doing"
                )
        return inputs

    def get_output_data(
        self, infer_results: tritonclient.http.InferResult
    ) -> Dict[str, np.ndarray]:
        """
        get output results as `np.ndarray`s from triton inference results

        Return
            Dict[str, np.ndarray]
                key is output name, value is its content
        """
        result = {}
        for output_struct in self.output_structs:
            name = output_struct["name"]
            dtype = output_struct['data_type']
            res = infer_results.as_numpy(name)
            if dtype == 'TYPE_STRING':
                res = np.array([i.decode() for i in res.reshape(res.size)])
            result[name] = res

        return result

    def forward(self, data_dict: Dict[str, np.ndarray], *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        inference steps for triton inference server, including:
        1. register input data: convert `np.ndarray` to `triton.http.inferInputs`
        2. model forward: call `tritonclient.infer`
        3. take out output data: convert `tritonclient.http.InferResult` to `dict[np.ndarray]`

        Arguments
            data_dict: Dict[str, np.ndarray]
                dictionary with network input data
        """
        infer_inputs = self.set_input_data(data_dict)
        infer_results = self.client.infer(
            model_name=self.model_name,
            inputs=infer_inputs,
        )
        outputs = self.get_output_data(infer_results)
        return outputs


class TritonInferenceModelSharedSystemMemory(TritonInferenceModel):
    """
    TritonInferenceModelSharedSystemMemory is similar to TritonInferenceModel.
    However, data transmission is through system shared memory.
    To utilize shared memory, input shape must be specified, hence size of shared memory could be registered correctly.
    If output shape is not provided, response will be sent through http.
    """

    def __init__(
        self,
        triton_url: str,
        model_name: str,
        model_version=str,
        input_shared_memory_prefix: str = "input",
        output_shared_memory_prefix: str = "output",
    ):
        super().__init__(triton_url, model_name, model_version)
        self.input_shared_memory_prefix = input_shared_memory_prefix
        self.output_shared_memory_prefix = output_shared_memory_prefix

        self.check_shared_memory_connection()
        self.infer_inputs = {}
        self.input_handles = {}
        self.infer_outputs = {}
        self.output_handles = {}

    def check_shared_memory_connection(self):
        """
        make sure shared memory connection is valid
        """
        try:
            self.client.unregister_system_shared_memory()
            item = np.zeros(1).astype(np.float32)
            byte_size = item.size * item.itemsize
            shm_input_handle = shm.create_shared_memory_region("tmp", "/tmp", byte_size)
            shm.set_shared_memory_region(shm_input_handle, [item])
            self.client.register_system_shared_memory("tmp", "/tmp", byte_size)
            shm.destroy_shared_memory_region(shm_input_handle)
        except Exception:
            raise ConnectionError("failed to connect to triton client with shared system memory.")

    def set_input_data(
        self, data_dict: Dict[str, np.ndarray]
    ) -> List[tritonclient.http.InferInput]:
        """
        turns input (`data_dict`) into triton input format.

        It is best to reuse previously registered shared memory buffers.
        However, if input shape is different to buffer size,
        buffer should be re-registered.

        Arguments
            data_dict: Dict[str, np.ndarray]
                key is input name, value is its content

        Returns
            List[tritonclient.http.InferInput]
                a list of triton inputs. order is not important.
        """
        for input_struct in self.input_structs:
            name = input_struct["name"]
            data_type = input_struct["data_type"]
            data_type = BaseTritonClient.DTYPES_CONFIG_TO_API[data_type]
            dims = input_struct["dims"]

            item = data_dict.get(name, None)
            if item is None:
                raise KeyError(f"cannot find required object {name} of dimension {dims}")

            item = item.astype(self.DTYPES_API_TO_NUMPY[data_type])

            if name not in self.infer_inputs or list(item.shape) != self.infer_inputs[name].shape():
                dims = list(item.shape)
                byte_size = item.size * item.itemsize

                if name in self.input_handles:
                    shm.destroy_shared_memory_region(self.input_handles[name])
                input_handle = shm.create_shared_memory_region(
                    name, f"/{self.input_shared_memory_prefix}_{name}", byte_size
                )
                self.client.unregister_system_shared_memory(name)
                self.client.register_system_shared_memory(
                    name, f"/{self.input_shared_memory_prefix}_{name}", byte_size
                )

                infer_input = tritonclient.http.InferInput(name, dims, data_type)
                infer_input.set_shared_memory(name, byte_size)

                self.infer_inputs[name] = infer_input
                self.input_handles[name] = input_handle

            shm.set_shared_memory_region(self.input_handles[name], [item])

        return list(self.infer_inputs.values())

    def set_output_data(
        self, output_shapes: Optional[Dict[str, List[int]]] = None
    ) -> List[tritonclient.http.InferInput]:
        """
        Similar logic to input data: we tend to reuse shared memory buffers unless shapes does not match.
        If `output_shapes` is not provided, we do not use share memory for outputs.
        """
        if not output_shapes:
            return None

        for output_struct in self.output_structs:
            name = output_struct["name"]
            data_type = output_struct["data_type"]
            data_type = BaseTritonClient.DTYPES_CONFIG_TO_API[data_type]
            dims = output_shapes.get(name, None)

            if dims is None:
                raise KeyError(f"cannot find dimension of required object {name}")

            if name not in self.infer_outputs or dims != self.infer_outputs[name].shape():
                item = np.zeros(dims)
                item = item.astype(self.DTYPES_API_TO_NUMPY[data_type])
                # TODO: handle byte string

                byte_size = item.size * item.itemsize
                output_handle = shm.create_shared_memory_region(
                    name, f"/{self.output_shared_memory_prefix}_{name}", byte_size
                )
                self.client.unregister_system_shared_memory(name)
                self.client.register_system_shared_memory(
                    name, f"/{self.output_shared_memory_prefix}_{name}", byte_size
                )

                infer_output = tritonclient.http.InferInput(name, dims, data_type)
                infer_output.set_shared_memory(name, byte_size)

                self.infer_outputs[name] = infer_output
                self.output_handles[name] = output_handle

        return list(self.infer_outputs.values())

    def get_output_data_shared_memory(
        self, infer_results: tritonclient.http.InferResult
    ) -> Dict[str, np.ndarray]:
        """
        Get the actual output data from shared memory.
        Note that we have to do `.copy()` to use the data.
        """
        result = {}
        for output_struct in self.output_structs:
            name = output_struct["name"]
            res = infer_results.get_output(name)
            if res:
                item = shm.get_contents_as_numpy(
                    self.output_handles[name],
                    tritonclient.utils.triton_to_np_dtype(res["datatype"]),
                    res["shape"],
                )
            result[name] = item.copy()

        return result

    def cleanup(self, handles: List[ctypes.c_void_p]):
        """
        Clean up shared memory buffers from both client and triton sides.
        """
        self.client.unregister_system_shared_memory()
        for handle in handles:
            shm.destroy_shared_memory_region(handle)

    def forward(
        self,
        data_dict: Dict[str, np.ndarray],
        output_shapes: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Arguments
            data_dict: dict[str, np.ndarray]
                dictionary with network input data
            output_shapes: Optional[Dict[str, List[int]]] = None
                to take advantage of shared memory in response, specify output shapes.
                key is name for each output struct, value is its shape in `list[int]`
        """
        infer_inputs = self.set_input_data(data_dict)
        infer_outputs = self.set_output_data(output_shapes)

        infer_results = self.client.infer(
            model_name=self.model_name,
            inputs=infer_inputs,
            outputs=infer_outputs,
        )

        is_shared_memory_response = infer_outputs is not None
        if is_shared_memory_response:
            outputs = self.get_output_data_shared_memory(infer_results)
        else:
            # if output shape is unknown, output will be sent through network to avoid memory allocation error
            outputs = self.get_output_data(infer_results)

        return outputs

    def __del__(self):
        self.cleanup(self.input_handles.values())
        self.cleanup(self.output_handles.values())
