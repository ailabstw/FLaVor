import json
import random
from json import JSONDecodeError
from typing import Any, Dict, List, Union

import cv2
import numpy as np
from nanoid import generate
from pydantic import TypeAdapter
from starlette.datastructures import FormData

from ..models import AiCOCOFormat, AiImage
from .base_strategy import BaseStrategy


class AiCOCOInputStrategy(BaseStrategy):
    async def apply(self, form_data: Union[FormData, Dict[str, Any]]):

        files = form_data.get("files")

        if "images" not in form_data:
            images = [
                {"id": generate(), "file_name": file, "index": idx}
                for idx, file in enumerate(files)
            ]
        else:
            ta = TypeAdapter(List[AiImage])
            try:
                images = json.loads(form_data.get("images"))
            except TypeError as e:
                raise JSONDecodeError(doc="", msg=str(e), pos=-1)

            ta.validate_python(images)

            for image in images:
                try:
                    image["file_name"] = next(file for file in files if image["file_name"] in file)
                except StopIteration:
                    raise Exception("filename not match")

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
        sorted_images: List[Any],
        categories: Dict[int, Any],
        seg_model_out: np.ndarray,
        meta: Dict[str, Any] = {},
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Reformat model output to AICOCO compatible format.

        Args:
            sorted_images (List[Any]): sort images in AICOCO
            categories (Dict[int, str]): Dictionary mapping class indices to its info.
            seg_model_out (np.ndarray): Segmentation output as a NumPy array.
            meta (Dict[str, Any]): Meta info for AICOCO.

        Returns:
            Dict[str, Any]: Result in AICOCO compatible format.
        """

        images = {"images": sorted_images}

        images_id_table = {idx: image["id"] for idx, image in enumerate(sorted_images)}

        categories = self.generate_categories(categories)

        class_id_table = {}
        for category in categories["categories"]:
            class_id = category.pop("class_id", None)
            display = category.pop("display", True)
            if display and class_id is not None:
                class_id_table[class_id] = category["id"]

        annot_obj = self.generate_annotations_objects(
            seg_model_out, images_id_table, class_id_table
        )

        meta = {"meta": meta}

        return {**images, **categories, **annot_obj, **meta}

    def generate_categories(self, categories: Dict[int, str]) -> Dict[str, Any]:

        res = dict()
        res["categories"] = list()
        supercategory_id_table = dict()

        for class_id, category in categories.items():
            if category.get("supercategory_name", None):
                if category["supercategory_name"] not in supercategory_id_table:
                    supercategory_id_table[category["supercategory_name"]] = generate()
            category["id"] = generate()
            category["class_id"] = class_id
            category["supercategory_id"] = supercategory_id_table.get(
                category.pop("supercategory_name", None), None
            )

            res["categories"].append(category)

        for sup_class_name, n_id in supercategory_id_table.items():
            res["categories"].append({"id": n_id, "name": sup_class_name, "supercategory_id": None})

        return res

    def generate_annotations_objects(
        self, volumn_4D: np.ndarray, images_id_table: Dict[int, str], class_id_table: Dict[int, str]
    ) -> Dict[str, Any]:
        """
        Generate annotations and objects in AICOCO compatible format.

        Args:
            volumn_4D (np.ndarray): 4D grouped volumetric data.
                The data should be preprocessed in connected regions with label index.
            images_id_table (Dict[int, str]): Dictionary mapping slice numbers to nanoid.
            class_id_table (Dict[int, str]): Dictionary mapping class indices to nanoid.

        Returns:
            Dict[str, Any]: A dictionary containing annotations and objects in AICOCO compatible format.
        """

        assert volumn_4D.ndim == 4, f"shape {volumn_4D.shape} is not 4D"  # class, z, x, y
        classes, slices, _, _ = volumn_4D.shape

        res = dict()
        res["annotations"] = list()
        res["objects"] = list()

        # Traverse classes
        for cls_idx in range(classes):
            if cls_idx not in class_id_table:
                continue

            class_nano_id = class_id_table[cls_idx]

            cls_volume = volumn_4D[cls_idx]
            unique_labels = np.unique(cls_volume)[1:]  # Ignore index 0
            label_nano_ids = {label_idx: generate() for label_idx in unique_labels}

            # Traverse 1~label
            for label_idx in unique_labels:

                label_nano_id = label_nano_ids[label_idx]

                # Traverse slices
                for slice_idx in range(slices):
                    label_slice = np.array(cls_volume[slice_idx])
                    image_nano_id = images_id_table[slice_idx]

                    the_label_slice = np.array(label_slice == label_idx, dtype=np.uint8)
                    if the_label_slice.sum() == 0:
                        continue

                    contours, _ = cv2.findContours(
                        the_label_slice,
                        cv2.RETR_TREE,
                        cv2.CHAIN_APPROX_NONE,  # No approximation
                    )

                    # Traverse contours
                    segmentation = list()
                    for _, contour in enumerate(contours):
                        _contour = contour.reshape(-1)
                        _contour = _contour.tolist()
                        segmentation.append(_contour)
                    res["annotations"].append(
                        {
                            "id": generate(),
                            "image_id": image_nano_id,
                            "object_id": label_nano_id,
                            "iscrowd": 0,
                            "bbox": None,
                            "segmentation": segmentation,
                        }
                    )
                res["objects"].append({"id": label_nano_id, "category_ids": [class_nano_id]})

        return res


class AiCOCOGradioStrategy(BaseStrategy):
    async def apply(self, result: Dict[str, Any]) -> List:
        data = np.transpose(result["data"], (1, 2, 3, 0))

        data = ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype(np.uint8)

        imgs = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.shape[2] == 1 else img for img in data]

        classes, slices, _, _ = result["seg_model_out"].shape

        # Traverse classes
        for cls_idx in range(classes):
            if not result["categories"][cls_idx]["display"]:
                continue

                hex_color = result["categories"][cls_idx].get(
                    "color", "#{:06x}".format(random.randint(0, 0xFFFFFF))
                )
                rgb_tuple = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))

            cls_volume = result["seg_model_out"][cls_idx]
            unique_labels = np.unique(cls_volume)[1:]  # Ignore index 0

            # Traverse 1~label
            for label_idx in unique_labels:

                # Traverse slices
                for slice_idx in range(slices):
                    label_slice = np.array(cls_volume[slice_idx])

                    the_label_slice = np.array(label_slice == label_idx, dtype=np.uint8)
                    if the_label_slice.sum() == 0:
                        continue

                    contours, _ = cv2.findContours(
                        the_label_slice,
                        cv2.RETR_TREE,
                        cv2.CHAIN_APPROX_NONE,  # No approximation
                    )

                    cv2.drawContours(imgs[slice_idx], contours, -1, rgb_tuple, 3)

        return imgs, None, "success"
