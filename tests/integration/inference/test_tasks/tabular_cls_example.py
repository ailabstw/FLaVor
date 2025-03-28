import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import cached_download, hf_hub_url

from flavor.serve.apps import InferAPP
from flavor.serve.inference.data_models.api import (
    AiCOCOTabularInputDataModel,
    AiCOCOTabularOutputDataModel,
)
from flavor.serve.inference.data_models.functional import AiTable
from flavor.serve.inference.inference_models import BaseAiCOCOTabularInferenceModel
from flavor.serve.inference.strategies import AiCOCOTabularClassificationOutputStrategy

REPO_ID = "julien-c/wine-quality"
FILENAME = "sklearn_model.joblib"


class ClassificationInferenceModel(BaseAiCOCOTabularInferenceModel):
    def __init__(self):
        super().__init__()
        self.formatter = AiCOCOTabularClassificationOutputStrategy()

    def define_inference_network(self) -> Callable:
        model = joblib.load(cached_download(hf_hub_url(REPO_ID, FILENAME)))
        return model

    def set_categories(self) -> List[Dict[str, Any]]:
        categories = [{"name": str(grade)} for grade in range(3, 9)]  # grade from 3 to 8
        return categories

    def set_regressions(self) -> None:
        return None

    def data_reader(self, files: Sequence[str], **kwargs) -> List[pd.DataFrame]:
        dataframes = [pd.read_csv(file) for file in files]
        return dataframes

    def preprocess(self, data: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(data)

    def inference(self, x: pd.DataFrame):
        out = self.network.predict(x).reshape(-1, 1)
        return out

    def postprocess(self, model_out: np.ndarray, **kwargs) -> np.ndarray:
        # Define the range of the model outputs
        min_value = 3
        max_value = 8

        # Number of possible output classes
        num_classes = max_value - min_value + 1

        # Flatten the model outputs to handle them
        model_out = model_out.flatten()

        # Ensure all model outputs are within the specified range
        if np.any(model_out < min_value) or np.any(model_out > max_value):
            raise ValueError("One or more model outputs are out of the expected range (3 to 8).")

        # Initialize the one-hot encoded array
        one_hot_batch = np.zeros((model_out.shape[0], num_classes), dtype=int)

        # Convert each model output to its corresponding one-hot encoded vector
        for i, output in enumerate(model_out):
            one_hot_batch[i, output - min_value] = 1

        return one_hot_batch

    def output_formatter(
        self,
        model_out: Any,
        tables: Sequence[AiTable],
        dataframes: Sequence[pd.DataFrame],
        meta: Dict[str, Any],
        categories: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AiCOCOTabularOutputDataModel:

        output = self.formatter(
            model_out=model_out,
            tables=tables,
            dataframes=dataframes,
            categories=categories,
            meta=meta,
        )
        return output


app = InferAPP(
    infer_function=ClassificationInferenceModel(),
    input_data_model=AiCOCOTabularInputDataModel,
    output_data_model=AiCOCOTabularOutputDataModel,
)

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 9111)))
