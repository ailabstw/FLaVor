import abc
import json

import numpy as np
import triton_python_backend_utils as pb_utils


class BaseTritonPythonModel(metaclass=abc.ABCMeta):
    """
    This class aims to simplify python backend model implementations on Triton inference servers.
    A valid python backend model must be name `TritonPythonModel`.
    You can inherit this `BaseTritonPythonModel` to handle the complex communication to Triton model server for you.

    A `TritonPythonModel` have three abstract methods:
    - initialize: called on model load. you should **initialize your model here**.
    - execute: handle requests
    - finalize: called on model unload

    A minimal example that echos the input values:

    ```python
    class TritonPythonModel(BaseTritonPythonModel):
        def forward(self, data_dict: dict):
            return data_dict
    ```
    """

    NP_BYTE_STRING = np.object_

    def initialize(self, args):
        """
        read devices and input/output structs
        """
        self.args = args
        self.model_config = json.loads(args["model_config"])

        self.input_configs = {}
        for inp in self.model_config.get("input", []):
            name = inp["name"]
            np_dtype = pb_utils.triton_string_to_numpy(inp["data_type"])
            self.input_configs[name] = {
                **inp,
                "np_dtype": np_dtype,
            }

        self.output_configs = {}
        for out in self.model_config.get("output", []):
            name = out["name"]
            np_dtype = pb_utils.triton_string_to_numpy(out["data_type"])
            self.output_configs[name] = {
                **out,
                "np_dtype": np_dtype,
            }

        self.device = "cuda" if self.args["model_instance_kind"] == "GPU" else "cpu"

    def get_input_objects(self, request) -> dict:
        """
        read input objects from request, decode as numpy object, and put them into dict
        """
        inputs = {}
        for name in self.input_configs:
            inp = pb_utils.get_input_tensor_by_name(request, name).as_numpy()
            np_dtype = self.input_configs[name]["np_dtype"]
            inp = inp.astype(np_dtype)
            if np_dtype == self.NP_BYTE_STRING:
                inp = [i.decode() for i in inp.reshape(inp.size)]
            inputs[name] = inp
        return inputs

    def set_output_objects(self, data_dict: dict):
        """
        set values of a model output (as dict) to valid triton response type
        """
        results = []
        for key, value in data_dict.items():
            if key not in self.output_configs:
                continue

            config = self.output_configs[key]

            value = np.array(value).astype(config["np_dtype"]).reshape(config["dims"])
            if self.output_configs[key]["np_dtype"] == self.NP_BYTE_STRING:
                value = np.array([v.encode() for v in value.reshape(value.size)])

            tensor = pb_utils.Tensor(key, value)
            results.append(tensor)

        return pb_utils.InferenceResponse(output_tensors=results)

    def execute(self, requests):
        responses = []
        for request in requests:
            response = self.pipeline(request)
            responses.append(response)
        return responses

    def pipeline(self, request):
        """
        This is an abstract method that can be overwritten.
        For example, you might want to convert inputs to `torch.tensor` if your model is a pytorch model.
        """
        data_dict = self.get_input_objects(request)
        out_dict = self.forward(data_dict)
        response = self.set_output_objects(out_dict)
        return response

    @abc.abstractmethod
    def forward(self, data_dict: dict):
        return NotImplemented
