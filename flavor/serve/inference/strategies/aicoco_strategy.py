from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2  # type: ignore
import numpy as np
from nanoid import generate  # type: ignore

from ..data_models.functional import (
    AiAnnotation,
    AiCategory,
    AiImage,
    AiMeta,
    AiObject,
    AiRegression,
    AiRegressionItem,
)
from .base_strategy import BaseStrategy

AiCOCOOut = Dict[
    str,
    Union[
        Sequence[AiImage],
        Sequence[AiCategory],
        Sequence[AiRegression],
        Sequence[AiAnnotation],
        Sequence[AiObject],
        AiMeta,
    ],
]
AiCOCORef = Dict[
    str, Union[Sequence[AiImage], Sequence[AiCategory], Sequence[AiRegression], AiMeta]
]


def check_any_nonint(x):
    return np.any(~(np.mod(x, 1) == 0))


class BaseAiCOCOOutputStrategy(BaseStrategy):
    @abstractmethod
    def model_to_aicoco(
        self, aicoco_ref: AiCOCOOut, model_out: Union[np.ndarray, Dict[str, Any]], **kwargs
    ) -> AiCOCOOut:
        """
        Abstract method to convert model output to AiCOCO compatible format.
        """
        raise NotImplementedError

    def generate_categories(self, categories: Sequence[Dict[str, Any]]) -> List[AiCategory]:
        """
        Generate `categories` field in AiCOCO compatible format.

        Args:
            categories (Sequence[Dict[str, Any]]): Dictionary mapping class indices to category nanoid.

        Returns:
            List[AiCategory]: List of `AiCategory` which is in AiCOCO compatible format.
        """
        res = list()
        supercategory_id_table = dict()
        self.class_id_table = dict()

        for class_id, category in enumerate(categories):
            supercategory_name = category.pop("supercategory_name", None)
            if supercategory_name:
                supercategory_id_table[supercategory_name] = supercategory_id_table.get(
                    supercategory_name, generate()
                )
            category["id"] = generate()
            category["supercategory_id"] = supercategory_id_table.get(supercategory_name)
            res.append(AiCategory.model_validate(category))
            if category.get("display", True):
                self.class_id_table[class_id] = category["id"]

        for sup_class_name, n_id in supercategory_id_table.items():
            supercategory = {"id": n_id, "name": sup_class_name, "supercategory_id": None}
            res.append(AiCategory.model_validate(supercategory))

        return res

    def generate_regressions(self, regressions: Sequence[Dict[str, Any]]) -> List[AiRegression]:
        """
        Generate `regressions` field in AiCOCO compatible format.

        Args:
            regressions (Sequence[Dict[str, Any]]): Dictionary mapping regression indices to regression nanoid.

        Returns:
            List[AiRegression]: List of `AiRegression` which is in AiCOCO compatible format.
        """
        res = list()
        superregression_id_table = dict()
        self.regression_id_table = dict()

        for regression_id, regression in enumerate(regressions):
            superregression_name = regression.pop("superregression_name", None)
            if superregression_name:
                superregression_id_table[superregression_name] = superregression_id_table.get(
                    superregression_name, generate()
                )
            regression["id"] = generate()
            regression["superregression_id"] = superregression_id_table.get(superregression_name)
            res.append(AiRegression.model_validate(regression))

            self.regression_id_table[regression_id] = regression["id"]

        for sup_regression_name, n_id in superregression_id_table.items():
            superregression = {"id": n_id, "name": sup_regression_name, "superregression_id": None}
            res.append(AiRegression.model_validate(superregression))

        return res

    def prepare_aicoco(
        self,
        images: Sequence[AiImage],
        categories: Optional[Sequence[Dict[str, Any]]] = None,
        regressions: Optional[Sequence[Dict[str, Any]]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[AiCOCORef, np.ndarray]:
        """
        Prepare prerequisite for AiCOCO.

        Args:
            images (Sequence[AiImage]): List of AiCOCO images field.
            categories (Optional[Sequence[Dict[str, Any]]]): List of unprocessed categories. Default: None.
            regressions (Optional[Sequence[Dict[str, Any]]]): List of unprocessed regressions. Default: None.
            meta (Optional[Dict[str, Any]]): AiCOCO meta field. Default: None.
        Returns:
            Tuple[AiCOCORef, np.ndarray]: Prepared AiCOCO output and inference model output array.
        """

        categories = categories if categories is not None else []
        regressions = regressions if regressions is not None else []
        meta = meta if meta is not None else {"category_ids": None, "regressions": None}

        if not hasattr(self, "aicoco_categories") and not hasattr(self, "aicoco_regressions"):
            # only activate at first run
            self.images_id_table = {idx: image.id for idx, image in enumerate(images)}
            self.aicoco_categories = self.generate_categories(categories)
            self.aicoco_regressions = self.generate_regressions(regressions)

        aicoco_ref = {
            "images": images,
            "categories": self.aicoco_categories,
            "regressions": self.aicoco_regressions,
            "meta": AiMeta(**meta),
        }

        return aicoco_ref


class AiCOCOSegmentationOutputStrategy(BaseAiCOCOOutputStrategy):
    def __call__(
        self,
        model_out: np.ndarray,
        images: Sequence[AiImage],
        categories: Sequence[Dict[str, Any]],
        **kwargs,
    ) -> AiCOCOOut:
        """
        Apply the AiCOCO output strategy to reformat the model's output.

        Args:
            model_out (np.ndarray): Inference model output.
            images (Sequence[AiImage]): List of AiCOCO images field.
            categories (Sequence[Dict[str, Any]]): List of unprocessed categories.

        Returns:
            AiCOCOOut: Result in AiCOCO compatible format.
        """
        self.validate_model_output(model_out)
        aicoco_ref = self.prepare_aicoco(images=images, categories=categories)
        aicoco_out = self.model_to_aicoco(aicoco_ref, model_out)
        return aicoco_out

    def validate_model_output(self, model_out: np.ndarray):
        """
        Validate inference model output.
        """
        if not isinstance(model_out, np.ndarray):
            raise TypeError(f"`model_out` must be type: np.ndarray but got {type(model_out)}.")

        if model_out.ndim != 3 and model_out.ndim != 4:
            raise ValueError(
                f"The dimension of `model_out` should be in 3D or 4D but got {model_out.ndim}."
            )

        if check_any_nonint(model_out):
            raise ValueError(
                "The value of `model_out` should be integer such as 0, 1, 2 ... with int or float type."
            )

    def model_to_aicoco(self, aicoco_ref: AiCOCORef, model_out: np.ndarray) -> AiCOCOOut:
        """
        Convert segmentation inference model output to AiCOCO compatible format.

        Args:
            aicoco_ref (AiCOCORef): Complete AiCOCO output.
            model_out (np.ndarray): Segmentation inference model output.

        Returns:
            AiCOCOOut: Result in AiCOCO compatible format.
        """
        annot_obj = self.generate_annotations_objects(model_out)

        return {**aicoco_ref, **annot_obj}

    def generate_annotations_objects(
        self, out: np.ndarray
    ) -> Dict[str, Union[Sequence[AiAnnotation], Sequence[AiObject]]]:
        """
        Generate `annotations` and `objects` in AiCOCO compatible format from 4D volumetric data.

        Args:
            out (np.ndarray): 3D or 4D predicted seg inference model output. This could be regular semantic seg mask or grouped instance seg mask.
        Returns:
            Dict[str, Union[Sequence[AiAnnotation], Sequence[AiObject]]]: Dictionary containing annotations and objects in AiCOCO compatible format.

        Notes:
            - The function assumes the input data is preprocessed with connected regions labeled with indices if it is an instance segmentation task.
            - Annotations are generated for each labeled region in each slice.
            - 'bbox' in annotations is set to None, and 'segmentation' is defined based on contours.
            - The 'iscrowd' field in annotations is set to 0 for non-crowd objects.
        """
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

                    annot = {
                        "id": generate(),
                        "image_id": image_nano_id,
                        "object_id": label_nano_id,
                        "iscrowd": 0,
                        "bbox": None,
                        "segmentation": segmentation,
                    }
                    res["annotations"].append(AiAnnotation(**annot))

                obj = {
                    "id": label_nano_id,
                    "category_ids": [class_nano_id],
                    "regressions": None,
                }
                res["objects"].append(AiObject(**obj))

        return res


class AiCOCOClassificationOutputStrategy(BaseAiCOCOOutputStrategy):
    def __call__(
        self,
        model_out: np.ndarray,
        images: Sequence[AiImage],
        categories: Sequence[Dict[str, Any]],
        **kwargs,
    ) -> AiCOCOOut:
        """
        Apply the AiCOCO output strategy to reformat the model's output.

        Args:
            model_out (np.ndarray): Inference model output.
            images (Sequence[AiImage]): List of AiCOCO images field.
            categories (Sequence[Dict[str, Any]]): List of unprocessed categories.

        Returns:
            AiCOCOOut: Result in AiCOCO compatible format.
        """
        self.validate_model_output(model_out)
        aicoco_ref = self.prepare_aicoco(images=images, categories=categories)
        aicoco_out = self.model_to_aicoco(aicoco_ref, model_out)
        return aicoco_out

    def validate_model_output(self, model_out: np.ndarray):
        """
        Validate inference model output.
        """
        if not isinstance(model_out, np.ndarray):
            raise TypeError(f"`model_out` must be type: np.ndarray but got {type(model_out)}.")

        if model_out.ndim != 1:
            raise ValueError(
                f"The dimension of `model_out` should be in 1D but got {model_out.ndim}."
            )

        if check_any_nonint(model_out):
            raise ValueError(
                "The value of `model_out` should be only 0 or 1 with int or float type."
            )

    def model_to_aicoco(
        self,
        aicoco_ref: AiCOCORef,
        model_out: np.ndarray,
    ) -> AiCOCOOut:
        """
        Convert classification inference model output to AiCOCO compatible format.

        Args:
            aicoco_ref (AiCOCORef): Complete AiCOCO output.
            model_out (np.ndarray): Classification inference model output.

        Returns:
            AiCOCOOut: Result in AiCOCO compatible format.
        """
        aicoco_ref["images"], aicoco_ref["meta"] = self.update_images_meta(
            model_out, aicoco_ref["images"], aicoco_ref["meta"]
        )

        annot_obj = {"annotations": [], "objects": []}

        return {**aicoco_ref, **annot_obj}

    def update_images_meta(
        self,
        out: np.ndarray,
        images: Sequence[AiImage],
        meta: AiMeta,
    ) -> Tuple[List[AiImage], AiMeta]:
        """
        Update `category_ids` in  `images` and `meta` based on the classification model output.

        Args:
            out (np.ndarray): Inference model output in 1D shape: (n,).
            images (Sequence[AiImage]): List of AiCOCO images field.
            meta (AiMeta): AiCOCO meta field.

        Returns:
            Tuple[List[AiImage], AiMeta]: Updated AiCOCO compatible images and meta field.

        Notes:
            The function updates the 'category_ids' field in both the images and meta dictionaries based on the inference model output.
            For 2D input data, it updates the last element of the images field list.
            For 3D input data, it updates meta field.
        """
        n_classes = len(out)

        assert n_classes == len(
            self.class_id_table
        ), f"The number of categories is not matched with the inference model output {out.shape}."

        for cls_idx in range(n_classes):
            if cls_idx not in self.class_id_table:
                raise ValueError(
                    f"Category {cls_idx} cannot be found. Please specify every category in counting numbers starting with 0."
                )
            class_nano_id = self.class_id_table[cls_idx]
            cls_pred = out[cls_idx]

            # For 2D case, handle `images`
            if len(self.images_id_table) == 1:
                if images[-1].category_ids is None:
                    images[-1].category_ids = list()
                if cls_pred:
                    images[-1].category_ids.append(class_nano_id)

            # For 3D case, handle `meta`
            else:
                if meta.category_ids is None:
                    meta.category_ids = list()
                if cls_pred:
                    meta.category_ids.append(class_nano_id)

        return images, meta


class AiCOCODetectionOutputStrategy(BaseAiCOCOOutputStrategy):
    def __call__(
        self,
        model_out: Dict[str, Any],
        images: Sequence[AiImage],
        categories: Sequence[Dict[str, Any]],
        regressions: Sequence[Dict[str, Any]],
        **kwargs,
    ) -> AiCOCOOut:
        """
        Apply the AiCOCO output strategy to reformat the model's output.

        Args:
            model_out (Dict[str, Any]): Inference model output.
            images (Sequence[AiImage]): List of AiCOCO images field.
            categories (Sequence[Dict[str, Any]]): List of unprocessed categories.
            regressions (Sequence[Dict[str, Any]]): List of unprocessed regressions.

        Returns:
            AiCOCOOut: Result in AiCOCO compatible format.
        """
        self.validate_model_output(model_out)
        aicoco_ref = self.prepare_aicoco(
            images=images, categories=categories, regressions=regressions
        )
        aicoco_out = self.model_to_aicoco(aicoco_ref, model_out)
        return aicoco_out

    def validate_model_output(self, model_out: Dict[str, Any]):
        """
        Validate inference model output.
        """
        if not isinstance(model_out, dict):
            raise TypeError("The type of inference model output should be `dict`.")
        if "bbox_pred" not in model_out:
            raise KeyError("Key `bbox_pred` must be in inference model output.")
        if "cls_pred" not in model_out:
            raise KeyError("Key `cls_pred` must be in inference model output.")

        bbox_pred = model_out.get("bbox_pred")
        cls_pred = model_out.get("cls_pred")
        confidence_score = model_out.get("confidence_score", None)
        regression_value = model_out.get("regression_value", None)

        if len(bbox_pred) != len(cls_pred):
            raise ValueError("`bbox_pred` and `cls_pred` should have same amount of elements.")

        if confidence_score is not None and len(bbox_pred) != len(confidence_score):
            raise ValueError(
                "`bbox_pred` and `confidence_score` should have same amount of elements."
            )

        if regression_value is not None and len(bbox_pred) != len(regression_value):
            raise ValueError(
                "`bbox_pred` and `regression_value` should have same amount of elements."
            )

        if not isinstance(cls_pred, np.ndarray) and not isinstance(cls_pred, list):
            raise TypeError(
                f"`cls_pred` must be type: np.ndarray or list but got {type(cls_pred)}."
            )

        if check_any_nonint(cls_pred):
            raise ValueError(
                "The value of `cls_pred` should be only 0 or 1 with int or float type."
            )

    def model_to_aicoco(
        self,
        aicoco_ref: AiCOCORef,
        model_out: Dict[str, Any],
    ) -> AiCOCOOut:
        """
        Convert detection inference model output to AiCOCO compatible format.

        Args:
            aicoco_ref (AiCOCORef): Complete AiCOCO output.
            model_out (Dict[str, Any]): Detection inference model output.

        Returns:
            AiCOCOOut: Result in AiCOCO compatible format.
        """
        annot_obj = self.generate_annotations_objects(model_out)

        return {**aicoco_ref, **annot_obj}

    def generate_annotations_objects(
        self,
        out: Dict[str, Any],
    ) -> Dict[str, Union[Sequence[AiAnnotation], Sequence[AiObject]]]:
        """
        Generate `annotations` and `objects` in AiCOCO compatible format from detection model output.

        Args:
            out (Dict[str, Any]): Detection inference model output dictionary with keys:
                - 'bbox_pred': List of bounding box predictions in the format [y_min, x_min, y_max, x_max].
                - 'cls_pred': List of one-hot classification result for each bbox.
                - 'confidence_score' (optional): List of confidence scores for each prediction.
                - 'regression_value' (optional): List of regression values for each prediction.

        Returns:
            Dict[str, Union[Sequence[AiAnnotation], Sequence[AiObject]]]: Dictionary containing annotations
                and objects in AiCOCO compatible format.

        Notes:
            - The function generates unique object IDs, image IDs, and annotation IDs using the `generate` function.
            - If 'confidence_score' is present in the output, it is added to the generated objects.
            - If 'regressions' are present in the output, they are added to the generated objects.
            - The 'bbox' in annotations is in the format [[x_min, y_min, x_max, y_max]].
            - The 'segmentation' in annotations is set to None for detection tasks.
        """
        res = dict()
        res["annotations"] = list()
        res["objects"] = list()

        for i, (bbox_pred, cls_pred) in enumerate(zip(out["bbox_pred"], out["cls_pred"])):
            if isinstance(bbox_pred, np.ndarray):
                bbox_pred = bbox_pred.tolist()
            y_min, x_min, y_max, x_max = bbox_pred

            image_nano_id = self.images_id_table[0]

            # handle objects
            object_nano_id = generate()
            obj = {
                "id": object_nano_id,
                "category_ids": [],
            }

            for c in range(len(cls_pred)):
                if cls_pred[c] == 0 or c not in self.class_id_table:
                    continue
                obj["category_ids"].append(self.class_id_table[c])

            if not obj["category_ids"]:
                # all the `display` flag in the predicted classes are false
                continue

            confidence_score = out.get("confidence_score")
            if confidence_score is not None:
                cs = confidence_score[i]
                obj["confidence"] = cs.item() if isinstance(cs, np.ndarray) else cs

            regression_value = out.get("regression_value")
            if regression_value is not None:
                obj["regressions"] = list()
                for i, value in enumerate(regression_value[i]):
                    regression_item = {
                        "regression_id": self.regression_id_table[i],
                        "value": value.item() if isinstance(value, np.ndarray) else value,
                    }
                    obj["regressions"].append(AiRegressionItem(**regression_item))
            else:
                obj["regressions"] = None

            res["objects"].append(AiObject(**obj))

            # handle annotations
            annot = {
                "id": generate(),
                "image_id": image_nano_id,
                "object_id": object_nano_id,
                "iscrowd": 0,
                "bbox": [[x_min, y_min, x_max, y_max]],
                "segmentation": None,
            }

            res["annotations"].append(AiAnnotation(**annot))

        return res


class AiCOCORegressionOutputStrategy(BaseAiCOCOOutputStrategy):
    def __call__(
        self,
        model_out: np.ndarray,
        images: Sequence[AiImage],
        regressions: Sequence[Dict[str, Any]],
        **kwargs,
    ) -> AiCOCOOut:
        """
        Apply the AiCOCO output strategy to reformat the model's output.

        Args:
            model_out (np.ndarray): Inference model output.
            images (Sequence[AiImage]): List of AiCOCO images field.
            regressions (Sequence[Dict[str, Any]]): List of unprocessed regressions.

        Returns:
            AiCOCOOut: Result in AiCOCO compatible format.
        """
        self.validate_model_output(model_out)
        aicoco_ref = self.prepare_aicoco(images=images, regressions=regressions)
        aicoco_out = self.model_to_aicoco(aicoco_ref, model_out)
        return aicoco_out

    def validate_model_output(self, model_out: np.ndarray):
        """
        Validate inference model output.
        """
        if not isinstance(model_out, np.ndarray):
            raise TypeError(f"`model_out` must be type: np.ndarray but got {type(model_out)}.")

        if model_out.ndim != 1 and model_out.ndim != 4:
            raise ValueError(
                f"The dimension of `model_out` should be in 1D but got {model_out.ndim}."
            )

    def model_to_aicoco(
        self,
        aicoco_ref: AiCOCORef,
        model_out: np.ndarray,
    ) -> AiCOCOOut:
        """
        Convert regression inference model output to AiCOCO compatible format.

        Args:
            aicoco_ref (AiCOCORef): Complete AiCOCO output.
            model_out (np.ndarray): Regression inference model output.

        Returns:
            AiCOCOOut: Result in AiCOCO compatible format.
        """
        aicoco_ref["images"], aicoco_ref["meta"] = self.update_images_meta(
            model_out, aicoco_ref["images"], aicoco_ref["meta"]
        )

        annot_obj = {"annotations": [], "objects": []}

        return {**aicoco_ref, **annot_obj}

    def update_images_meta(
        self,
        out: np.ndarray,
        images: Sequence[AiImage],
        meta: AiMeta,
    ) -> Tuple[List[AiImage], AiMeta]:
        """
        Update `regressions` in `images` and `meta` based on the regression model output.

        Args:
            out (np.ndarray): Inference model output in 1D shape: (n,).
            images (Sequence[AiImage]): List of AiCOCO images field.
            meta (AiMeta): AiCOCO meta field.

        Returns:
            Tuple[List[AiImage], AiMeta]: Updated AiCOCO compatible images and meta field.

        Notes:
            The function updates the 'regressions' field in both the images and meta dictionaries based on the model output.
            For 2D input data, it updates the last element of the images field list.
            For 3D input data, it updates meta.
        """
        n_regression = len(out)

        assert n_regression == len(
            self.regression_id_table
        ), f"The number of regression values is not matched with the inference model output {out.shape}."

        for reg_idx in range(n_regression):
            if reg_idx not in self.regression_id_table:
                raise ValueError(
                    f"Regression {reg_idx} cannot be found. Please specify every regression value in counting numbers starting with 0."
                )
            pred_value = out[reg_idx].item()
            regression_nano_id = self.regression_id_table[reg_idx]

            # For 2D case, handle `images`
            if len(self.images_id_table) == 1:
                if images[-1].regressions is None:
                    images[-1].regressions = list()
                regression_item = {
                    "regression_id": regression_nano_id,
                    "value": pred_value,
                }
                images[-1].regressions.append(AiRegressionItem(**regression_item))

            # For 3D case, handle `meta`
            else:
                if meta.regressions is None:
                    meta.regressions = list()
                regression_item = {
                    "regression_id": regression_nano_id,
                    "value": pred_value,
                }
                meta.regressions.append(AiRegressionItem(**regression_item))

        return images, meta
