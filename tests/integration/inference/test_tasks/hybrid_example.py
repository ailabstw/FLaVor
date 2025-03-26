import os
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from flavor.serve.apps import InferAPP
from flavor.serve.inference.data_models.api import (
    AiCOCOHybridInputDataModel,
    AiCOCOHybridOutputDataModel,
)
from flavor.serve.inference.data_models.functional import AiImage, AiTable
from flavor.serve.inference.inference_models import BaseAiCOCOHybridInferenceModel
from flavor.serve.inference.strategies import AiCOCOHybridClassificationOutputStrategy

torch.manual_seed(1234)
np.random.seed(1234)


class HybridNet(nn.Module):
    def __init__(self, csv_input_dim, image_feature_dim, fusion_hidden_dim=128):
        """
        Args:
            csv_input_dim (int): csv dimension
            image_feature_dim (int): image feature dimension
            fusion_hidden_dim (int): fusion hidden dimension
        """
        super(HybridNet, self).__init__()

        # image branch
        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.image_fc = nn.Linear(64, image_feature_dim)

        # tabular branch
        self.csv_fc = nn.Linear(csv_input_dim, image_feature_dim)

        # fusion branch
        self.fusion_fc = nn.Linear(2 * image_feature_dim, fusion_hidden_dim)

        # classifier
        self.out = nn.Linear(fusion_hidden_dim, 2)

    def forward(self, tensor_image, tensor_tabular):
        """
        Args:
            tensor_image (tensor)
            tensor_tabular (tensor)
        Returns:
            logits (tensor)
        """
        # image forward
        x_img = self.image_conv(tensor_image)  # shape: (batch_size, 64, 1, 1)
        x_img = x_img.view(x_img.size(0), -1)  # shape: (batch_size, 64)
        x_img = self.image_fc(x_img)  # shape: (batch_size, image_feature_dim)

        # tabular forward
        x_csv = self.csv_fc(tensor_tabular)  # shape: (batch_size, image_feature_dim)

        # fusion
        x = torch.cat((x_img, x_csv), dim=1)  # shape: (batch_size, 2 * image_feature_dim)
        x = F.relu(self.fusion_fc(x))
        logits = self.out(x)
        return logits


class ClassificationInferenceModel(BaseAiCOCOHybridInferenceModel):
    def __init__(self):
        super().__init__()
        self.formatter = AiCOCOHybridClassificationOutputStrategy()

    def define_inference_network(self) -> Callable:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = HybridNet(csv_input_dim=22, image_feature_dim=32)
        network.eval()
        network.to(self.device)
        return network

    def set_categories(self) -> List[Dict[str, Any]]:
        categories = [{"name": str(i)} for i in range(2)]
        return categories

    def set_regressions(self) -> None:
        return None

    def data_reader(self, image_files: Sequence[str], table_files: Sequence[str]):
        image = Image.open(image_files[0])
        tabular = pd.read_csv(table_files[0])
        return image, tabular

    def preprocess(self, x):
        image_data, table_data = x
        # image data
        img = np.array(image_data).transpose(2, 0, 1).astype(np.float32)
        tensor_image = torch.tensor(img).unsqueeze(0).to(self.device)

        # csv data
        tabular = table_data
        tensor_tabular = torch.tensor(tabular.values.astype(np.float32)).to(self.device)

        return tensor_image, tensor_tabular

    def inference(self, x) -> torch.Tensor:
        tensor_image, tensor_tabular = x
        with torch.no_grad():
            out = self.network(tensor_image, tensor_tabular)
        return out

    def postprocess(self, model_out: torch.Tensor, **kwargs) -> np.ndarray:
        model_out = model_out.squeeze(0).cpu().detach()
        model_out = (nn.functional.softmax(model_out, dim=0) > 0.4).long()
        return model_out.numpy()

    def output_formatter(
        self,
        model_out: np.ndarray,
        images: Sequence[AiImage],
        tables: Sequence[AiTable],
        categories: Sequence[Dict[str, Any]],
        **kwargs
    ) -> AiCOCOHybridOutputDataModel:

        output = self.formatter(
            model_out=model_out, images=images, tables=tables, categories=categories
        )
        return output


app = InferAPP(
    infer_function=ClassificationInferenceModel(),
    input_data_model=AiCOCOHybridInputDataModel,
    output_data_model=AiCOCOHybridOutputDataModel,
)

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 9111)))
