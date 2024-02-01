import abc
import random
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .base_strategy import BaseStrategy


class BaseGradioStrategy(BaseStrategy):
    @abc.abstractmethod
    async def apply(self, *args, **kwargs):
        raise NotImplementedError

    def generate_rgb(self):
        components = ["r", "g", "b"]
        random.shuffle(components)

        rgb = {}
        for component in components:
            if component == components[0]:
                rgb[component] = random.randint(0, 255)
            elif component == components[1]:
                rgb[component] = random.randint(158, 255)
            else:
                rgb[component] = random.randint(0, 98)

        return rgb["r"], rgb["g"], rgb["b"]


class SegmentationGradioStrategy(BaseGradioStrategy):
    async def apply(self, result: Dict[str, Any]) -> Tuple[List[Any], List[Any], None, str]:

        data = result["data"]  # shape: (c, z, y, x) or (c, y, x)
        data = ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype(np.uint8)
        if data.ndim == 3:
            data = np.expand_dims(data, axis=1)
        data = np.repeat(data, 3, axis=0) if data.shape[0] == 1 else data

        mask = np.zeros_like(data)

        for cls_idx in range(result["model_out"].shape[0]):
            if not result["categories"][cls_idx]["display"]:
                continue

            if "color" in result["categories"][cls_idx]:
                rgb_tuple = tuple(
                    int(result["categories"][cls_idx]["color"][i : i + 2], 16) for i in (1, 3, 5)
                )
            else:
                rgb_tuple = self.generate_rgb()

            cls_volume = result["model_out"][cls_idx]

            mask[0][cls_volume != 0] = rgb_tuple[0]
            mask[1][cls_volume != 0] = rgb_tuple[1]
            mask[2][cls_volume != 0] = rgb_tuple[2]

        pred_vis = (data * 0.8 + mask * 0.2).astype(np.uint8)

        data = np.transpose(data, (1, 2, 3, 0))
        pred_vis = np.transpose(pred_vis, (1, 2, 3, 0))

        return [img for img in data], [img for img in pred_vis], None, "success"


class DetectionGradioStrategy(BaseGradioStrategy):
    async def apply(self, result: Dict[str, Any]) -> Tuple[List[Any], List[Any], None, str]:

        data = result["data"]  # shape: (c, y, x)
        data = ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype(np.uint8)
        data = np.repeat(data, 3, axis=0) if data.shape[0] == 1 else data

        img = Image.fromarray(data.transpose((1, 2, 0))).convert("RGB")
        draw = ImageDraw.Draw(img)

        bbox_pred = result["model_out"]["bbox_pred"]
        cls_pred = result["model_out"]["cls_pred"]
        confidence_score = (
            result["model_out"]["confidence_score"]
            if "confidence_score" in result["model_out"]
            else None
        )

        for i in range(len(bbox_pred)):
            y_min, x_min, y_max, x_max = bbox_pred[i]
            cls_idx = np.argmax(cls_pred[i])
            confidence = confidence_score[i] if confidence_score is not None else ""

            if "color" in result["categories"][cls_idx]:
                rgb_tuple = tuple(
                    int(result["categories"][cls_idx]["color"][i : i + 2], 16) for i in (1, 3, 5)
                )
            else:
                rgb_tuple = self.generate_rgb()

            name = result["categories"][cls_idx]["name"]
            draw.rectangle(tuple([x_min, y_min, x_max, y_max]), outline=rgb_tuple, width=1)
            draw.text((x_min, y_min), f"{name} {confidence:.4f}", fill=rgb_tuple, align="left")

        data = np.transpose(data, (1, 2, 0))
        bbox_vis = np.array(img)

        return [data], [bbox_vis], None, "success"
