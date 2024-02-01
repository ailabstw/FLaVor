import abc
import copy
import json
from json import JSONDecodeError
from typing import Any, Dict, List, Tuple, Union

import cv2  # type: ignore
import numpy as np
from nanoid import generate  # type: ignore
from pydantic import TypeAdapter
from starlette.datastructures import FormData

from ..models import AiCOCOFormat, AiImage
from .base_strategy import BaseStrategy


class AiCOCOInputStrategy(BaseStrategy):
    async def apply(self, form_data: Union[FormData, Dict[str, Union[List[str], str]]]):
        """
        Apply the AiCOCO input strategy to process input data.

        Args:
            form_data (Union[FormData, Dict[str, Union[List[str], str]]]): Input data in the form of FormData or a dictionary.

        Returns:
            Dict[str, Any]: Processed data in AiCOCO compatible `images` format.
        """
        files = form_data.get("files")

        if "images" not in form_data:
            images = [
                {
                    "id": generate(),
                    "file_name": file,
                    "physical_file_name": file,
                    "index": idx,
                    "category_ids": None,
                    "regressions": None,
                }
                for idx, file in enumerate(files)
            ]
        else:
            images = self.load_and_validate(form_data, "images", List[AiImage])
            for image in images:
                image["physical_file_name"] = self.match_file_name(image["file_name"], files)

        return {"images": images}

    def load_and_validate(self, form_data, key, data_model):

        try:
            data = json.loads(form_data.get(key))
        except JSONDecodeError as e:
            raise JSONDecodeError(doc="", msg=str(e), pos=-1)
        ta = TypeAdapter(data_model)
        ta.validate_python(data)

        return data

    def match_file_name(self, file_name, files):

        try:
            return next(file for file in files if file_name in file)
        except StopIteration:
            raise Exception(f"Filename {file_name} not match")


class AiCOCOOutputStrategy(BaseStrategy):
    async def apply(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the AiCOCO output strategy to reformat the model's output.

        Args:
            **result (Dict[str, Any]): A dictionary containing the model's output.

        Returns:
            Dict[str, Any]: Result in complete AICOCO format.
        """

        ta = TypeAdapter(AiCOCOFormat)

        aicoco_out, model_out = self.prepare_aicoco(**result)

        response = self.model_to_aicoco(aicoco_out, model_out)

        ta.validate_python(response)

        return response

    @abc.abstractmethod
    def model_to_aicoco(self, *args, **kwargs):
        """
        Abstract method to convert model output to AiCOCO compatible format.
        """
        raise NotImplementedError

    def generate_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate `categories` in AiCOCO compatible format by removing physical_file_name.

        Args:
            images (List[Dict[str, Any]]): List of dictionary mapping for AiCOCO image attribute.

        Returns:
            List[Dict[str, Any]]: Modified list of dictionary mapping for AiCOCO image attribute.
        """
        for image in images:
            if "physical_file_name" in image:
                image.pop("physical_file_name", None)

        return images

    def generate_categories(self, categories: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate `categories` in AiCOCO compatible format.

        Args:
            categories (Dict[int, Dict[str, Any]]): Dictionary mapping class indices to category nanoid.

        Returns:
            List[Dict[str, Any]]: Dictionary containing AiCOCO compatible `categories` format.
        """
        res = list()
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

            res.append(category)

        for sup_class_name, n_id in supercategory_id_table.items():
            res.append({"id": n_id, "name": sup_class_name, "supercategory_id": None})

        return res

    def generate_regressions(self, regressions: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate `regressions` in AiCOCO compatible format.

        Args:
            regressions (Dict[int, Dict[str, Any]]): Dictionary mapping regression indices to regression nanoid.

        Returns:
            List[Dict[str, Any]]: Dictionary containing AiCOCO compatible `regressions` format.
        """
        res = list()
        superregression_id_table = dict()

        for regression_id, regression in regressions.items():
            if regression.get("superregression_name", None):
                if regression["superregression_name"] not in superregression_id_table:
                    superregression_id_table[regression["superregression_name"]] = generate()
            regression["id"] = generate()
            regression["regression_id"] = regression_id
            regression["superregression_id"] = superregression_id_table.get(
                regression.pop("superregression_name", None), None
            )

            res.append(regression)

        for sup_regression_name, n_id in superregression_id_table.items():
            res.append({"id": n_id, "name": sup_regression_name, "superregression_id": None})

        return res

    def set_images_class_regression_id_table(
        self,
        images: List[Dict[str, Any]],
    ) -> None:
        """
        Set table attributes for mapping image, class, and regression IDs.

        Args:
            images (List[Dict[str, Any]]): List of images.
        """
        self.images_id_table = {idx: image["id"] for idx, image in enumerate(images)}
        self.class_id_table = {}
        self.regression_id_table = {}
        for category in self.aicoco_categories:
            class_id = category.pop("class_id", None)
            display = category["display"] if "display" in category else True
            if display and class_id is not None:
                self.class_id_table[class_id] = category["id"]

        for regression in self.aicoco_regressions:
            regression_id = regression.pop("regression_id", None)
            if regression_id is not None:
                self.regression_id_table[regression_id] = regression["id"]

    def prepare_aicoco(
        self,
        model_out: np.ndarray,
        sorted_images: List[Dict[str, Any]],
        categories: Dict[int, Any],
        regressions: Dict[int, Any],
        meta: Dict[str, Any] = {"category_ids": None, "regressions": None},
        **kwargs,
    ) -> Tuple[Dict[str, Any], Union[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Prepare prerequisite for AiCOCO.

        Args:
            model_out (np.ndarray): Model output.
            sorted_images (List[Dict[str, Any]]): Sorted images.
            categories (Dict[int, Any]): Dictionary of unprocessed categories.
            regressions (Dict[int, Any]): Dictionary of unprocessed regressions.
            meta (Dict[str, Any]): Dictionary of meta. Default: `{"category_ids": None, "regressions": None}`.

        Returns:
            Tuple[Dict[str, Any], Union[np.ndarray, Dict[str, np.ndarray]]]: Prepared AiCOCO output and model output as np.ndarray.
        """
        aicoco_images = {"images": self.generate_images(copy.deepcopy(sorted_images))}

        if not hasattr(self, "aicoco_categories") and not hasattr(self, "aicoco_regressions"):
            self.aicoco_categories = self.generate_categories(copy.deepcopy(categories))
            self.aicoco_regressions = self.generate_regressions(copy.deepcopy(regressions))
            self.set_images_class_regression_id_table(sorted_images)

        aicoco_categories = {"categories": self.aicoco_categories}
        aicoco_regressions = {"regressions": self.aicoco_regressions}
        aicoco_meta = {"meta": copy.deepcopy(meta)}

        return {
            **aicoco_images,
            **aicoco_categories,
            **aicoco_regressions,
            **aicoco_meta,
        }, model_out


class AiCOCOSegmentationOutputStrategy(AiCOCOOutputStrategy):
    def model_to_aicoco(
        self,
        aicoco_out: Dict[str, Any],
        model_out: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Convert segmentation model output to AiCOCO compatible format.

        Args:
            aicoco_out (Dict[str, Any]): AiCOCO output.
            model_out (np.ndarray): Segmentation model output.

        Returns:
            Dict[str, Any]: Result in AiCOCO compatible format.
        """
        annot_obj = self.generate_annotations_objects(model_out)

        return {**aicoco_out, **annot_obj}

    def generate_annotations_objects(self, out: np.ndarray) -> Dict[str, Any]:
        """
        Generate `annotations` and `objects` in AiCOCO compatible format from 4D volumetric data.

        Args:
            out (np.ndarray): 3D or 4D predicted seg model output. This could be regular semantic seg mask or grouped instance seg mask.
        Returns:
            Dict[str, Any]: Dictionary containing annotations and objects in AiCOCO compatible format.

        Notes:
            - The function assumes the input data is preprocessed with connected regions labeled with indices if it is an instance segmentation task.
            - Annotations are generated for each labeled region in each slice.
            - 'bbox' in annotations is set to None, and 'segmentation' is defined based on contours.
            - The 'iscrowd' field in annotations is set to 0 for non-crowd objects.
        """

        assert out.ndim == 3 or out.ndim == 4, f"shape {out.shape} is not in 3D or 4D"

        if out.ndim == 3:
            out = np.expand_dims(out, axis=1)

        classes, slices = out.shape[:2]

        res = dict()
        res["annotations"] = list()
        res["objects"] = list()

        # Traverse classes
        for cls_idx in range(classes):
            if cls_idx not in self.class_id_table:
                continue

            class_nano_id = self.class_id_table[cls_idx]

            cls_volume = out[cls_idx]
            unique_labels = np.unique(cls_volume)[1:]  # Ignore index 0
            label_nano_ids = {label_idx: generate() for label_idx in unique_labels}

            # Traverse 1~label
            for label_idx in unique_labels:

                label_nano_id = label_nano_ids[label_idx]

                # Traverse slices
                for slice_idx in range(slices):
                    label_slice = np.array(cls_volume[slice_idx])
                    image_nano_id = self.images_id_table[slice_idx]

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
                res["objects"].append(
                    {"id": label_nano_id, "category_ids": [class_nano_id], "regressions": None}
                )

        return res


class AiCOCOClassificationOutputStrategy(AiCOCOOutputStrategy):
    def model_to_aicoco(
        self,
        aicoco_out: Dict[str, Any],
        model_out: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Convert classification model output to AiCOCO compatible format.

        Args:
            aicoco_out (Dict[str, Any]): AiCOCO output.
            model_out (np.ndarray): Classification model output.

        Returns:
            Dict[str, Any]: Result in AiCOCO compatible format.
        """
        aicoco_out["images"], aicoco_out["meta"] = self.update_images_meta(
            model_out, aicoco_out["images"], aicoco_out["meta"]
        )

        annot_obj = {"annotations": [], "objects": []}

        return {**aicoco_out, **annot_obj}

    def update_images_meta(
        self,
        out: np.ndarray,
        images: List[Dict[str, Any]],
        meta: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Update `category_ids` in  `images` and `meta` based on the classification model output.

        Args:
            out (np.ndarray): Model output in 1D shape: (n,).
            images (List[Dict[str, Any]]): List of image metadata dictionaries.
            meta (Dict[str, Any]): Meta information dictionary.

        Returns:
            Tuple[List[Dict[str, Any]], Dict[str, Any]]: Updated AiCOCO compatible images and meta dictionaries.

        Notes:
            The function updates the 'category_ids' field in both the images and meta dictionaries based on the model output.
            For 2D input data, it updates both meta and the last image in the images list.
            For 3D input data, it only updates meta.
        """
        assert out.ndim == 1, f"shape {out.shape} is not 1D"
        n_classes = len(out)

        assert n_classes == len(self.class_id_table), "Number of categories is not matched."

        for cls_idx in range(n_classes):
            if cls_idx not in self.class_id_table:
                raise ValueError(f"class {cls_idx} not found. Please specify every category.")
            class_nano_id = self.class_id_table[cls_idx]
            cls_pred = out[cls_idx]

            # For 2D case, handle `images`
            if len(self.images_id_table) == 1:
                if images[-1]["category_ids"] is None:
                    images[-1]["category_ids"] = list()
                if cls_pred:
                    images[-1]["category_ids"].append(class_nano_id)

            # For 3D case, handle `meta`
            else:
                if meta["category_ids"] is None:
                    meta["category_ids"] = list()
                if cls_pred:
                    meta["category_ids"].append(class_nano_id)

        return images, meta


class AiCOCODetectionOutputStrategy(AiCOCOOutputStrategy):
    def model_to_aicoco(
        self,
        aicoco_out: Dict[str, Any],
        model_out: Dict[str, List[np.ndarray]],
    ) -> Dict[str, Any]:
        """
        Convert detection model output to AiCOCO compatible format.

        Args:
            aicoco_out (Dict[str, Any]): AiCOCO output.
            model_out (Dict[str, List[np.ndarray]]): Detection model output.

        Returns:
            Dict[str, Any]: Result in AiCOCO compatible format.
        """
        annot_obj = self.generate_annotations_objects(model_out)

        return {**aicoco_out, **annot_obj}

    def generate_annotations_objects(
        self,
        out: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate `annotations` and `objects` in AiCOCO compatible format from detection model output.

        Args:
            out (Dict[str, Any]): Detection model output dictionary with keys:
                - 'bbox_pred': List of bounding box predictions in the format [y_min, x_min, y_max, x_max].
                - 'cls_pred': List of one-hot classification result for each bbox.
                - 'confidence_score' (optional): List of confidence scores for each prediction.
                - 'regression_value' (optional): List of regression values for each prediction.

        Returns:
            Dict[str, Any]: Dictionary containing annotations and objects in AiCOCO compatible format.

        Notes:
            - The function generates unique object IDs, image IDs, and annotation IDs using the `generate` function.
            - If 'confidence_score' is present in the output, it is added to the generated objects.
            - If 'regressions' are present in the output, they are added to the generated objects.
            - The 'bbox' in annotations is in the format [[x_min, y_min, x_max, y_max]].
            - The 'segmentation' in annotations is set to None for detection tasks.
        """
        assert isinstance(out, dict), "`out` type should be dict."

        res = dict()
        res["annotations"] = list()
        res["objects"] = list()

        for i, bbox_pred in enumerate(out["bbox_pred"]):
            y_min, x_min, y_max, x_max = bbox_pred.tolist()

            image_nano_id = self.images_id_table[0]

            # handle objects
            object_nano_id = generate()
            obj = {
                "id": object_nano_id,
                "category_ids": [],
            }

            cls_pred = out["cls_pred"][i]
            for c in range(len(cls_pred)):
                if cls_pred[c] == 0 or c not in self.class_id_table:
                    continue
                obj["category_ids"].append(self.class_id_table[c])

            if not obj["category_ids"]:
                # all the `display` flag in the predicted classes are false
                continue

            if "confidence_score" in out:
                obj["confidence"] = out["confidence_score"][i].item()

            if "regressions" in out:
                obj["regressions"] = list()
                regression_value = out["regression_value"][i]
                for i, value in enumerate(regression_value):
                    obj["regressions"].append(
                        {
                            "regression_id": self.regression_id_table[i],
                            "value": value.item(),
                        }
                    )
            else:
                obj["regressions"] = None

            res["objects"].append(obj)

            # handle annotations
            annot = {
                "id": generate(),
                "image_id": image_nano_id,
                "object_id": object_nano_id,
                "iscrowd": 0,
                "bbox": [[x_min, y_min, x_max, y_max]],
                "segmentation": None,
            }

            res["annotations"].append(annot)

        return res


class AiCOCORegressionOutputStrategy(AiCOCOOutputStrategy):
    def model_to_aicoco(
        self,
        aicoco_out: Dict[str, Any],
        model_out: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Convert regression model output to AiCOCO compatible format.

        Args:
            aicoco_out (Dict[str, Any]): AiCOCO output.
            model_out (np.ndarray): Regression model output.

        Returns:
            Dict[str, Any]: Result in AiCOCO compatible format.
        """
        aicoco_out["images"], aicoco_out["meta"] = self.update_images_meta(
            model_out, aicoco_out["images"], aicoco_out["meta"]
        )

        annot_obj = {"annotations": [], "objects": []}

        return {**aicoco_out, **annot_obj}

    def update_images_meta(
        self,
        out: np.ndarray,
        images: List[Dict[str, Any]],
        meta: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Update `regressions` in `images` and `meta` based on the regression model output.

        Args:
            out (np.ndarray): Model output in 1D shape: (n,).
            images (List[Dict[str, Any]]): List of image metadata dictionaries.
            meta (Dict[str, Any]): Meta information dictionary.

        Returns:
            Tuple[List[Dict[str, Any]], Dict[str, Any]]: Updated AiCOCO compatible images and meta dictionaries.

        Notes:
            The function updates the 'regressions' field in both the images and meta dictionaries based on the model output.
            For 2D input data, it updates both meta and the last image in the images list.
            For 3D input data, it only updates meta.
        """
        assert out.ndim == 1, f"shape {out.shape} is not 1D"

        n_regression = len(out)

        assert n_regression == len(
            self.regression_id_table
        ), "Number of regressions is not matched."

        for reg_idx in range(n_regression):
            if reg_idx not in self.regression_id_table:
                raise ValueError(
                    f"class {reg_idx} not found. Please specify every regression category."
                )
            pred_value = out[reg_idx].item()
            regression_nano_id = self.regression_id_table[reg_idx]

            # For 2D case, handle `images`
            if len(self.images_id_table) == 1:
                if images[-1]["regressions"] is None:
                    images[-1]["regressions"] = list()
                images[-1]["regressions"].append(
                    {
                        "regression_id": regression_nano_id,
                        "value": pred_value,
                    }
                )

            # For 3D case, handle `meta`
            else:
                if meta["regressions"] is None:
                    meta["regressions"] = list()
                meta["regressions"].append(
                    {
                        "regression_id": regression_nano_id,
                        "value": pred_value,
                    }
                )

        return images, meta
