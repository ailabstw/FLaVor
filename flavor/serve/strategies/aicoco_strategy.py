import json
from json import JSONDecodeError
from typing import Any, Dict, List, Union

from pydantic import TypeAdapter
from starlette.datastructures import FormData

from ..models import AiImage, AiCOCOFormat
from .base_strategy import BaseStrategy

from skimage.measure import label
from nanoid import generate
import cv2
import numpy as np


class AiCOCOInputStrategy(BaseStrategy):
    async def apply(self, form_data: Union[FormData, Dict[str, Any]]):

        ta = TypeAdapter(List[AiImage])
        try:
            images = json.loads(form_data.get("images"))
        except TypeError as e:
            raise JSONDecodeError(doc="", msg=str(e), pos=-1)

        ta.validate_python(images)

        files = form_data.get("files")

        for image, file in zip(images, files):
            image["physical_file_name"] = file

        return {"images": images}


class AiCOCOOutputStrategy(BaseStrategy):

    async def apply(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the AiCOCO output strategy to reformat the model's output.

        Args:
            **result (Dict[str, Any]): A dictionary containing the model's output.

        Returns:
            Dict[str, Any]: Result in AICOCO compatible format.
        """
        ta = TypeAdapter(AiCOCOFormat)

        response = self.model_to_aicoco(**result)

        ta.validate_python(response)

        return response

    def model_to_aicoco(
        self,
        model_out: np.ndarray,
        images_id_table: Dict[int, str],
        class_id_table: Dict[int, str],
        input_json: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Reformat model output to AICOCO compatible format.

        Args:
            model_out (np.ndarray): Model output as a NumPy array.
            images_id_table (Dict[int, str]): Dictionary mapping slice numbers to nanoid.
            class_id_table (Dict[int, str]): Dictionary mapping class indices to nanoid.
            input_json (Dict[str, Any]): Input JSON decoupled from the request.

        Returns:
            Dict[str, Any]: Result in AICOCO compatible format.
        """
        annot = self.generate_annotations_objects(model_out, images_id_table, class_id_table)
        return {**annot, **input_json}

    def generate_annotations_objects(self, volumn_4D, images_id_table, class_id_table) -> Dict[str, Any]:
        """
        Generate annotations and objects in AICOCO compatible format.

        Args:
            volumn_4D (np.ndarray): 4D grouped volumetric data. 
                The data should be precessed in connected regions with label index.
            images_id_table (Dict[int, str]): Dictionary mapping slice numbers to nanoid.
            class_id_table (Dict[int, str]): Dictionary mapping class indices to nanoid.
            back_ground_idx (int, optional): Index of the background class. Defaults to 0.

        Returns:
            Dict[str, Any]: A dictionary containing annotations and objects in AICOCO compatible format.
        """

        assert volumn_4D.ndim == 4, f"shape {volumn_4D.shape} is not 4D"  # class, z, x, y
        classes, slices, h, w = volumn_4D.shape

        res = dict()
        res["annotations"] = list()
        res["objects"] = list()

        # traverse classes
        for cls_idx in range(classes):
            if cls_idx not in class_id_table:
                continue

            nid_cls = class_id_table[cls_idx]

            cls_volumn = volumn_4D[cls_idx]
            label_num = np.unique(cls_volumn)[1:]  # ignore index 0

            # traverse slices
            for slice_idx in range(slices):
                label_slice = np.array(cls_volumn[slice_idx])
                nid_slice = images_id_table[slice_idx]

                # travser 1~label
                for label_idx in label_num:
                    the_label_slice = np.array(label_slice == label_idx, dtype=np.uint8)
                    if the_label_slice.sum() == 0:
                        continue
                    nid_label = generate()

                    contours, _ = cv2.findContours(
                        the_label_slice,
                        cv2.RETR_TREE,
                        cv2.CHAIN_APPROX_NONE,  # no approx
                    )

                    # traverse contours
                    segmentation = list()
                    for idx, contour in enumerate(contours):
                        _contour = contour.reshape(-1)
                        _contour = _contour.tolist()
                        segmentation.append(_contour)
                    res["annotations"].append(
                        {
                            "id": generate(),
                            "image_id": nid_slice,
                            "object_id": nid_label,
                            "iscrowd": 0,
                            "bbox": None,
                            "segmentation": segmentation,
                        }
                    )
                    res["objects"].append({"id": nid_label, "category_ids": [nid_cls]})

        return res
