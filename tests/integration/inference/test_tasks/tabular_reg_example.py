import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from flavor.serve.apps import InferAPP
from flavor.serve.inference.data_models.api import (
    BaseAiCOCOTabularInputDataModel,
    BaseAiCOCOTabularOutputDataModel,
)
from flavor.serve.inference.data_models.functional import AiTable
from flavor.serve.inference.inference_models import BaseAiCOCOTabularInferenceModel
from flavor.serve.inference.strategies import AiCOCOTabularRegressionOutputStrategy

torch.manual_seed(1234)
np.random.seed(1234)


class SimpleRegressor(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32, output_dim: int = 1):
        super(SimpleRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


class RegressionInferenceModel(BaseAiCOCOTabularInferenceModel):
    def __init__(self):
        self.formatter = AiCOCOTabularRegressionOutputStrategy()
        super().__init__()

    def define_inference_network(self) -> nn.Module:
        input_dim = 8  # change this if needed
        model = SimpleRegressor(input_dim=input_dim, hidden_dim=32, output_dim=1)
        model.eval()  # Set the model to evaluation mode.
        return model

    def set_categories(self) -> None:
        return None

    def set_regressions(self) -> List[Dict[str, Any]]:
        regressions = [{"name": "reg_value"}]
        return regressions

    def data_reader(
        self, tables: Dict[str, Any], files: Sequence[str], **kwargs
    ) -> List[pd.DataFrame]:
        dataframes = [pd.read_csv(file) for file in files]
        return dataframes

    def preprocess(self, data: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(data)

    def inference(self, x: pd.DataFrame):
        with torch.no_grad():
            input_tensor = torch.tensor(x.values.astype(np.float32))
            output_tensor = self.network(input_tensor)
            out = output_tensor.numpy().reshape(-1, 1)
        return out

    def postprocess(self, model_out: np.ndarray, **kwargs) -> np.ndarray:
        return model_out

    def output_formatter(
        self,
        model_out: Any,
        tables: Sequence[AiTable],
        dataframes: Sequence[pd.DataFrame],
        meta: Dict[str, Any],
        regressions: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs,
    ) -> BaseAiCOCOTabularOutputDataModel:

        output = self.formatter(
            model_out=model_out,
            tables=tables,
            dataframes=dataframes,
            regressions=regressions,
            meta=meta,
        )
        return output


app = InferAPP(
    infer_function=RegressionInferenceModel(),
    input_data_model=BaseAiCOCOTabularInputDataModel,
    output_data_model=BaseAiCOCOTabularOutputDataModel,
)

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 9111)))
