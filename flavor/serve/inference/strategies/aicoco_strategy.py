import json
import os
import random
import string
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import cv2  # type: ignore
import numpy as np
import pandas as pd
from nanoid import generate as nanoid_generate
from pydantic import BaseModel

from ..data_models.api.inference_data_model import (
    AiCOCOHybridOutputDataModel,
    AiCOCOImageOutputDataModel,
    AiCOCOTabularOutputDataModel,
)
from ..data_models.functional import (
    AiAnnotation,
    AiCategory,
    AiHybridMeta,
    AiImage,
    AiMeta,
    AiObject,
    AiRegression,
    AiRegressionItem,
    AiTable,
    AiTableMeta,
)
from .base_strategy import BaseStrategy

# --------------------------------------------------------------------------------
# Note: The following section may cause conflicts in multi-threaded environments.
# Please consider removing or replacing it with another solution if this is a concern.
# --------------------------------------------------------------------------------
GLOBAL_SEED = None


def set_global_seed(seed):
    """
    Sets a global random seed. GLOBAL_SEED will be updated each time generate() is called.
    May cause unpredictable behavior in multi-threaded environments or repeated calls. Use with caution.
    """
    global GLOBAL_SEED
    GLOBAL_SEED = seed


def generate(size=21):
    """
    Generates a random string ID.
    If GLOBAL_SEED is not None, it seeds the random generator on each call
    and increments GLOBAL_SEED by 1. Otherwise, it directly calls nanoid_generate().
    """

    global GLOBAL_SEED
    if GLOBAL_SEED is not None:
        random.seed(GLOBAL_SEED)
        GLOBAL_SEED += 1
        return "".join(random.choices(string.ascii_letters + string.digits + "_-", k=size))
    else:
        return nanoid_generate()


class AiCOCORef(BaseModel):
    images: List[AiImage]
    categories: List[AiCategory]
    regressions: List[AiRegression]
    meta: Union[AiMeta, AiTableMeta, AiHybridMeta]


class AiCOCOAnnotObj(TypedDict):
    annotations: List[AiAnnotation]
    objects: List[AiObject]


def check_any_nonint(x):
    return np.any(~(np.mod(x, 1) == 0))


class BaseAiCOCOOutputStrategy(BaseStrategy):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def model_to_aicoco(self, aicoco_ref: AiCOCORef, model_out: Any) -> AiCOCOImageOutputDataModel:
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
        res: List[AiCategory] = []
        supercategory_id_table: Dict[str, str] = {}

        for cat_dict in categories:
            cat_copy = cat_dict.copy()

            supercat_name = cat_copy.get("supercategory_name")

            if supercat_name and supercat_name not in supercategory_id_table:
                supercategory_id_table[supercat_name] = generate()

            cat_copy["id"] = generate()
            cat_copy["supercategory_id"] = supercategory_id_table.get(supercat_name)

            cat_copy.pop("supercategory_name", None)

            res.append(AiCategory.model_validate(cat_copy))

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
        res: List[AiRegression] = []
        superregression_id_table: Dict[str, str] = {}

        for reg_dict in regressions:
            reg_copy = reg_dict.copy()

            super_name = reg_copy.get("superregression_name")
            if super_name and super_name not in superregression_id_table:
                superregression_id_table[super_name] = generate()

            reg_copy["id"] = generate()
            reg_copy["superregression_id"] = superregression_id_table.get(super_name)

            reg_copy.pop("superregression_name", None)

            res.append(AiRegression.model_validate(reg_copy))

        for sup_regression_name, n_id in superregression_id_table.items():
            superregression = {
                "id": n_id,
                "name": sup_regression_name,
                "superregression_id": None,
            }
            res.append(AiRegression.model_validate(superregression))

        return res

    def prepare_aicoco(
        self,
        images: List[AiImage],
        categories: Optional[Sequence[Dict[str, Any]]] = None,
        regressions: Optional[Sequence[Dict[str, Any]]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> AiCOCORef:
        """
        Prepare prerequisite for AiCOCO.

        Args:
            images (List[AiImage]): List of AiCOCO images field.
            categories (Optional[Sequence[Dict[str, Any]]]): List of unprocessed categories. Default: None.
            regressions (Optional[Sequence[Dict[str, Any]]]): List of unprocessed regressions. Default: None.
            meta (Optional[Dict[str, Any]]): AiCOCO meta field. Default: None.
        Returns:
            AiCOCORef: Prepared AiCOCO output and inference model output array.
        """

        categories = categories if categories is not None else []
        regressions = regressions if regressions is not None else []
        meta = meta if meta is not None else {"category_ids": None, "regressions": None}
        aicoco_categories = self.generate_categories(categories)
        aicoco_regressions = self.generate_regressions(regressions)

        aicoco_ref = {
            "images": images,
            "categories": aicoco_categories,
            "regressions": aicoco_regressions,
            "meta": meta,
        }

        return AiCOCORef.model_validate(aicoco_ref)


class AiCOCOSegmentationOutputStrategy(BaseAiCOCOOutputStrategy):
    def __call__(
        self,
        model_out: np.ndarray,
        images: List[AiImage],
        categories: Sequence[Dict[str, Any]],
        **kwargs,
    ) -> AiCOCOImageOutputDataModel:
        """
        Apply the AiCOCO output strategy to reformat the model's output.

        Args:
            model_out (np.ndarray): Inference model output.
            images (List[AiImage]): List of AiCOCO images field.
            categories (Sequence[Dict[str, Any]]): List of unprocessed categories.

        Returns:
            AiCOCOImageOutputDataModel: Result in AiCOCOImageOutputDataModel.
        """
        self.validate_model_output(model_out, categories)
        aicoco_ref = self.prepare_aicoco(images=images, categories=categories)
        aicoco_out = self.model_to_aicoco(aicoco_ref, model_out)
        return aicoco_out

    def validate_model_output(self, model_out: np.ndarray, categories: Sequence[Dict[str, Any]]):
        """
        Validate inference model output.
        """
        if not isinstance(model_out, np.ndarray):
            raise TypeError(f"`model_out` must be type: np.ndarray but got {type(model_out)}.")

        if len(model_out) != len(categories):
            raise ValueError(
                f"The number of classes in `model_out` should be {len(categories)} "
                f"but got {len(model_out)}."
            )

        if model_out.ndim not in (3, 4):
            raise ValueError(
                f"The dimension of `model_out` should be in 3D or 4D but got {model_out.ndim}."
            )

        if check_any_nonint(model_out):
            raise ValueError("The value of `model_out` should be integer (0,1,2...) with int type.")

    def model_to_aicoco(
        self, aicoco_ref: AiCOCORef, model_out: np.ndarray
    ) -> AiCOCOImageOutputDataModel:
        """
        Convert segmentation inference model output to AiCOCO compatible format.

        Args:
            aicoco_ref (AiCOCORef): Complete AiCOCO output.
            model_out (np.ndarray): Segmentation inference model output.

        Returns:
            AiCOCOImageOutputDataModel: Result in AiCOCOImageOutputDataModel.
        """
        annot_obj = self.generate_annotations_objects(
            model_out,
            aicoco_ref.categories,
            aicoco_ref.images,
        )

        aicoco_ref = aicoco_ref.model_dump()

        return AiCOCOImageOutputDataModel.model_validate({**aicoco_ref, **annot_obj})

    def generate_annotations_objects(
        self,
        out: np.ndarray,
        categories: List[AiCategory],
        images: List[AiImage],
    ) -> AiCOCOAnnotObj:
        """
        Generate `annotations` and `objects` in AiCOCO compatible format from 4D volumetric data.

        Args:
            out (np.ndarray): 3D or 4D predicted seg inference model output. This could be regular semantic seg mask or grouped instance seg mask.
            categories (Sequence[Dict[str, Any]]): List of categories.
            images (List[AiImage]): List of AiCOCO images field.
        Returns:
            Dict[str, Union[Sequence[AiAnnotation], Sequence[AiObject]]]: Dictionary containing annotations and objects in AiCOCO compatible format.

        Notes:
            - The function assumes the input data is preprocessed with connected regions labeled with indices if it is an instance segmentation task.
            - Annotations are generated for each labeled region in each slice.
            - 'bbox' in annotations is set to None, and 'segmentation' is defined based on contours.
            - The 'iscrowd' field in annotations is set to 0 for non-crowd objects.
        """
        if out.ndim == 3:
            out = np.expand_dims(out, axis=1)  # shape => (class, slices, H, W)

        classes, slices = out.shape[:2]
        images_ids = [img.id for img in images]

        res = {
            "annotations": [],
            "objects": [],
        }

        # Traverse classes
        for cls_idx in range(classes):
            if not getattr(categories[cls_idx], "display", True):
                continue

            class_nano_id = categories[cls_idx].id
            cls_volume = out[cls_idx]
            unique_labels = np.unique(cls_volume)[1:]  # ignore 0
            label_nano_ids = {label_idx: generate() for label_idx in unique_labels}

            # Traverse 1~label
            for label_idx in unique_labels:

                label_nano_id = label_nano_ids[label_idx]

                # Traverse slices
                for slice_idx in range(slices):
                    label_slice = np.array(cls_volume[slice_idx])
                    image_nano_id = images_ids[slice_idx]

                    the_label_slice = np.array(label_slice == label_idx, dtype=np.uint8)
                    if the_label_slice.sum() == 0:
                        continue

                    contours, _ = cv2.findContours(
                        the_label_slice,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )

                    # Traverse contours
                    segmentation = []
                    for contour in contours:
                        _contour = contour.reshape(-1).tolist()
                        segmentation.append(_contour)

                    annot = AiAnnotation(
                        id=generate(),
                        image_id=image_nano_id,
                        object_id=label_nano_id,
                        iscrowd=0,
                        bbox=None,
                        segmentation=segmentation,
                    )
                    res["annotations"].append(annot)

                obj = AiObject(
                    id=label_nano_id,
                    category_ids=[class_nano_id],
                    regressions=None,
                )
                res["objects"].append(obj)

        return res


class AiCOCOClassificationOutputStrategy(BaseAiCOCOOutputStrategy):
    def __call__(
        self,
        model_out: np.ndarray,
        images: List[AiImage],
        categories: Sequence[Dict[str, Any]],
        **kwargs,
    ) -> AiCOCOImageOutputDataModel:
        """
        Apply the AiCOCO output strategy to reformat the model's output.

        Args:
            model_out (np.ndarray): Inference model output.
            images (List[AiImage]): List of AiCOCO images field.
            categories (Sequence[Dict[str, Any]]): List of unprocessed categories.

        Returns:
            AiCOCOImageOutputDataModel: Result in AiCOCOImageOutputDataModel.
        """
        self.validate_model_output(model_out, categories)
        aicoco_ref = self.prepare_aicoco(images=images, categories=categories)
        aicoco_out = self.model_to_aicoco(aicoco_ref, model_out)
        return aicoco_out

    def validate_model_output(self, model_out: np.ndarray, categories: Sequence[Dict[str, Any]]):
        """
        Validate inference model output.
        """
        if not isinstance(model_out, np.ndarray):
            raise TypeError(f"`model_out` must be type: np.ndarray but got {type(model_out)}.")

        if len(model_out) != len(categories):
            raise ValueError(
                f"The number of classes in `model_out` should be {len(categories)} "
                f"but got {len(model_out)}."
            )

        if model_out.ndim != 1:
            raise ValueError(f"The dimension of `model_out` should be 1D but got {model_out.ndim}.")

        if check_any_nonint(model_out):
            raise ValueError(
                "The value of `model_out` should be only 0 or 1 with int or float type."
            )

    def model_to_aicoco(
        self,
        aicoco_ref: AiCOCORef,
        model_out: np.ndarray,
    ) -> AiCOCOImageOutputDataModel:
        """
        Convert classification inference model output to AiCOCO compatible format.

        Args:
            aicoco_ref (AiCOCORef): Complete AiCOCO output.
            model_out (np.ndarray): Classification inference model output.

        Returns:
            AiCOCOImageOutputDataModel: Result in AiCOCOImageOutputDataModel.
        """
        updated_images, updated_meta = self.update_images_meta(
            model_out, aicoco_ref.images, aicoco_ref.meta, aicoco_ref.categories
        )

        annot_obj = {"annotations": [], "objects": []}

        aicoco_out = {
            "images": updated_images,
            "categories": aicoco_ref.categories,
            "regressions": aicoco_ref.regressions,
            "meta": updated_meta,
            **annot_obj,
        }

        return AiCOCOImageOutputDataModel.model_validate(aicoco_out)

    def update_images_meta(
        self,
        out: np.ndarray,
        images: List[AiImage],
        meta: AiMeta,
        categories: List[AiCategory],
    ) -> Tuple[List[AiImage], AiMeta]:
        """
        Update `category_ids` in  `images` and `meta` based on the classification model output.

        Args:
            out (np.ndarray): Inference model output in 1D shape: (n,).
            images (List[AiImage]): List of AiCOCO images field.
            meta (AiMeta): AiCOCO meta field.
            categories (Sequence[Dict[str, Any]]): List of categories.

        Returns:
            Tuple[List[AiImage], AiMeta]: Updated AiCOCO compatible images and meta field.

        Notes:
            The function updates the 'category_ids' field in both the images and meta dictionaries based on the inference model output.
            For 2D input data, it updates the last element of the images field list.
            For 3D input data, it updates meta field.
        """
        n_classes = len(out)

        for cls_idx in range(n_classes):
            cls_pred = out[cls_idx]
            category_id = categories[cls_idx].id

            # For 2D case -> images
            if len(images) == 1:
                if images[-1].category_ids is None:
                    images[-1].category_ids = []
                if cls_pred:
                    images[-1].category_ids.append(category_id)
            else:
                # For 3D case -> meta
                if meta.category_ids is None:
                    meta.category_ids = []
                if cls_pred:
                    meta.category_ids.append(category_id)

        return images, meta


class AiCOCODetectionOutputStrategy(BaseAiCOCOOutputStrategy):
    def __call__(
        self,
        model_out: Dict[str, Any],
        images: List[AiImage],
        categories: Sequence[Dict[str, Any]],
        regressions: Sequence[Dict[str, Any]],
        **kwargs,
    ) -> AiCOCOImageOutputDataModel:
        """
        Apply the AiCOCO output strategy to reformat the model's output.

        Args:
            model_out (Dict[str, Any]): Inference model output.
            images (List[AiImage]): List of AiCOCO images field.
            categories (Sequence[Dict[str, Any]]): List of unprocessed categories.
            regressions (Sequence[Dict[str, Any]]): List of unprocessed regressions.

        Returns:
            AiCOCOImageOutputDataModel: Result in AiCOCOImageOutputDataModel.
        """
        self.validate_model_output(model_out, categories)
        aicoco_ref = self.prepare_aicoco(
            images=images, categories=categories, regressions=regressions
        )
        aicoco_out = self.model_to_aicoco(aicoco_ref, model_out)
        return aicoco_out

    def validate_model_output(
        self, model_out: Dict[str, Any], categories: Sequence[Dict[str, Any]]
    ):
        """
        Validate inference model output.
        """
        if not isinstance(model_out, dict):
            raise TypeError("The type of inference model output should be `dict`.")
        if "bbox_pred" not in model_out:
            raise KeyError("Key `bbox_pred` must be in inference model output.")
        if "cls_pred" not in model_out:
            raise KeyError("Key `cls_pred` must be in inference model output.")

        bbox_pred = model_out["bbox_pred"]
        cls_pred = model_out["cls_pred"]
        confidence_score = model_out.get("confidence_score", None)
        regression_value = model_out.get("regression_value", None)

        if len(bbox_pred) != len(cls_pred):
            raise ValueError("`bbox_pred` and `cls_pred` should have same length.")

        if confidence_score is not None and len(bbox_pred) != len(confidence_score):
            raise ValueError("`bbox_pred` and `confidence_score` should have same length.")

        if regression_value is not None and len(bbox_pred) != len(regression_value):
            raise ValueError("`bbox_pred` and `regression_value` should have same length.")

        if not isinstance(cls_pred, (np.ndarray, list)):
            raise TypeError(f"`cls_pred` must be np.ndarray or list but got {type(cls_pred)}.")

        for pred in cls_pred:
            if len(pred) != len(categories):
                raise ValueError(
                    f"Each element in `cls_pred` should have length {len(categories)}."
                )

        if check_any_nonint(cls_pred):
            raise ValueError(
                "The value of `cls_pred` should be only 0 or 1 with int or float type."
            )

    def model_to_aicoco(
        self,
        aicoco_ref: AiCOCORef,
        model_out: Dict[str, Any],
    ) -> AiCOCOImageOutputDataModel:
        """
        Convert detection inference model output to AiCOCO compatible format.

        Args:
            aicoco_ref (AiCOCORef): Complete AiCOCO output.
            model_out (Dict[str, Any]): Detection inference model output.

        Returns:
            AiCOCOImageOutputDataModel: Result in AiCOCOImageOutputDataModel.
        """
        annot_obj = self.generate_annotations_objects(
            model_out,
            aicoco_ref.categories,
            aicoco_ref.regressions,
            aicoco_ref.images,
        )
        aicoco_out = {**aicoco_ref.model_dump(), **annot_obj}

        return AiCOCOImageOutputDataModel.model_validate(aicoco_out)

    def generate_annotations_objects(
        self,
        out: Dict[str, Any],
        categories: List[AiCategory],
        regressions: List[AiRegression],
        images: List[AiImage],
    ) -> AiCOCOAnnotObj:
        """
        Generate `annotations` and `objects` in AiCOCO compatible format from detection model output.

        Args:
            out (Dict[str, Any]): Detection inference model output dictionary with keys:
                - 'bbox_pred': List of bounding box predictions in the format [x_min, y_min, x_max, y_max].
                - 'cls_pred': List of one-hot classification result for each bbox.
                - 'confidence_score' (optional): List of confidence scores for each prediction.
                - 'regression_value' (optional): List of regression values for each prediction.
            images (List[AiImage]): List of AiCOCO images field.
            categories (Sequence[Dict[str, Any]]): List of categories.
            regressions (Sequence[Dict[str, Any]]): List of regressions.

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
        res = {"annotations": [], "objects": []}

        # detection: only support single image (2D)
        image_nano_id = images[0].id

        for i, (bbox_pred, cls_pred) in enumerate(zip(out["bbox_pred"], out["cls_pred"])):

            # handle objects
            object_nano_id = generate()
            obj = {
                "id": object_nano_id,
                "category_ids": [],
                "regressions": [] if "regression_value" in out else None,
            }

            # category (multi-label support)
            for cls_idx, val in enumerate(cls_pred):
                if val == 1 and getattr(categories[cls_idx], "display", True):
                    obj["category_ids"].append(categories[cls_idx].id)

            if not obj["category_ids"]:
                continue

            # confidence_score
            confidence_score = out.get("confidence_score")
            if confidence_score is not None:
                cs = confidence_score[i]
                cs_val = cs.item() if isinstance(cs, np.ndarray) else cs
                obj["confidence"] = cs_val

            # regression
            regression_value = out.get("regression_value")
            if regression_value is not None:
                for value, regression in zip(regression_value[i], regressions):
                    val = value.item() if isinstance(value, np.ndarray) else value
                    reg_item = AiRegressionItem(regression_id=regression.id, value=val)
                    obj["regressions"].append(reg_item)

            res["objects"].append(AiObject(**obj))

            # handle annotations
            if isinstance(bbox_pred, np.ndarray):
                bbox_pred = bbox_pred.tolist()
            x_min, y_min, x_max, y_max = bbox_pred

            annot = AiAnnotation(
                id=generate(),
                image_id=image_nano_id,
                object_id=object_nano_id,
                iscrowd=0,
                bbox=[[x_min, y_min, x_max, y_max]],
                segmentation=None,
            )
            res["annotations"].append(annot)

        return res


class AiCOCORegressionOutputStrategy(BaseAiCOCOOutputStrategy):
    def __call__(
        self,
        model_out: np.ndarray,
        images: List[AiImage],
        regressions: Sequence[Dict[str, Any]],
        **kwargs,
    ) -> AiCOCOImageOutputDataModel:
        """
        Apply the AiCOCO output strategy to reformat the model's output.

        Args:
            model_out (np.ndarray): Inference model output.
            images (List[AiImage]): List of AiCOCO images field.
            regressions (Sequence[Dict[str, Any]]): List of unprocessed regressions.

        Returns:
            AiCOCOImageOutputDataModel: Result in AiCOCOImageOutputDataModel.
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
                f"The dimension of `model_out` should be 1D or 4D but got {model_out.ndim}."
            )

    def model_to_aicoco(
        self,
        aicoco_ref: AiCOCORef,
        model_out: np.ndarray,
    ) -> AiCOCOImageOutputDataModel:
        """
        Convert regression inference model output to AiCOCO compatible format.

        Args:
            aicoco_ref (AiCOCORef): Complete AiCOCO output.
            model_out (np.ndarray): Regression inference model output.

        Returns:
            AiCOCOImageOutputDataModel: Result in AiCOCOImageOutputDataModel.
        """
        updated_images, updated_meta = self.update_images_meta(
            model_out, aicoco_ref.images, aicoco_ref.meta, aicoco_ref.regressions
        )
        annot_obj = {"annotations": [], "objects": []}

        aicoco_out = {
            "images": updated_images,
            "categories": aicoco_ref.categories,
            "regressions": aicoco_ref.regressions,
            "meta": updated_meta,
            **annot_obj,
        }
        return AiCOCOImageOutputDataModel.model_validate(aicoco_out)

    def update_images_meta(
        self,
        out: np.ndarray,
        images: List[AiImage],
        meta: AiMeta,
        regressions: List[AiRegression],
    ) -> Tuple[List[AiImage], AiMeta]:
        """
        Update `regressions` in `images` and `meta` based on the regression model output.

        Args:
            out (np.ndarray): Inference model output in 1D shape: (n,).
            images (List[AiImage]): List of AiCOCO images field.
            meta (AiMeta): AiCOCO meta field.
            regressions (Sequence[Dict[str, Any]]): List of regressions.

        Returns:
            Tuple[List[AiImage], AiMeta]: Updated AiCOCO compatible images and meta field.

        Notes:
            The function updates the 'regressions' field in both the images and meta dictionaries based on the model output.
            For 2D input data, it updates the last element of the images field list.
            For 3D input data, it updates meta.
        """
        n_regression = len(out)

        for reg_idx in range(n_regression):
            reg_pred = out[reg_idx].item() if isinstance(out[reg_idx], np.ndarray) else out[reg_idx]
            regression_id = regressions[reg_idx].id

            # 2D image
            if len(images) == 1:
                if images[-1].regressions is None:
                    images[-1].regressions = []
                reg_item = AiRegressionItem(regression_id=regression_id, value=reg_pred)
                images[-1].regressions.append(reg_item)
            else:
                # 3D -> meta
                if meta.regressions is None:
                    meta.regressions = []
                reg_item = AiRegressionItem(regression_id=regression_id, value=reg_pred)
                meta.regressions.append(reg_item)

        return images, meta


class BaseAiCOCOTabularOutputStrategy(BaseAiCOCOOutputStrategy):
    @abstractmethod
    def model_to_aicoco(
        self, aicoco_ref: Dict[str, Any], model_out: np.ndarray, **kwargs
    ) -> AiCOCOTabularOutputDataModel:
        """
        Abstract method to convert model output to AiCOCO compatible format.
        """
        raise NotImplementedError

    def iter_record_windows(
        self, dataframes: Sequence[pd.DataFrame], tables: Sequence[AiTable], meta: Dict[str, Any]
    ):
        """Yield table/window metadata for tabular records without materialising records."""
        window_size = meta["window_size"]
        for df, table in zip(dataframes, tables):
            num_records = len(df) // window_size
            for i in range(num_records):
                yield table, list(range(i * window_size, (i + 1) * window_size))

    def write_records_file(
        self, records, records_output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        output_dir = Path(
            records_output_dir
            or os.environ.get("AICOCO_RECORDS_OUTPUT_DIR")
            or tempfile.gettempdir()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"records_{generate()}.jsonl"
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        rows = 0

        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                    f.write("\n")
                    rows += 1
            tmp_path.replace(output_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        return {
            "format": "jsonl",
            "path": str(output_path),
            "rows": rows,
            "bytes": output_path.stat().st_size,
        }

    def prepare_aicoco(
        self,
        tables: Sequence[Dict[str, Any]],
        meta: Dict[str, Any],
        dataframes: Sequence[pd.DataFrame],
        categories: Optional[Sequence[Dict[str, Any]]] = None,
        regressions: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Prepare tabular AiCOCO metadata without creating in-memory records."""
        aicoco_tables = [AiTable(**tb) for tb in tables]
        categories = categories if categories is not None else []
        regressions = regressions if regressions is not None else []

        return {
            "tables": aicoco_tables,
            "categories": self.generate_categories(categories),
            "regressions": self.generate_regressions(regressions),
            "meta": meta,
        }


class AiCOCOTabularClassificationOutputStrategy(BaseAiCOCOTabularOutputStrategy):
    def __call__(
        self,
        model_out: np.ndarray,
        tables: Sequence[Dict[str, Any]],
        dataframes: Sequence[pd.DataFrame],
        categories: Sequence[Dict[str, Any]],
        meta: Dict[str, Any],
        **kwargs,
    ) -> AiCOCOTabularOutputDataModel:
        """
        Apply the AiCOCO output strategy to reformat the model's output.

        Args:
            model_out (np.ndarray): Inference model output.
            tables (Sequence[Dict[str, Any]]): List of AiCOCO table compatible dict.
            dataframes (Sequence[pd.DataFrame]): Sequence of dataframes correspond each tables.
            categories (Sequence[Dict[str, Any]]): List of unprocessed categories.
            meta (Dict[str, Any]): Additional metadata.

        Returns:
            AiCOCOOut: Result in AiCOCO compatible format.
        """
        self.validate_model_output(model_out, categories)
        aicoco_ref = self.prepare_aicoco(
            tables=tables, meta=meta, dataframes=dataframes, categories=categories
        )
        aicoco_out = self.model_to_aicoco(
            aicoco_ref,
            model_out,
            dataframes=dataframes,
            records_output_dir=kwargs.get("records_output_dir"),
        )
        return aicoco_out

    def validate_model_output(self, model_out: np.ndarray, categories: Sequence[Dict[str, Any]]):
        """
        Validate inference model output.
        """
        if not isinstance(model_out, np.ndarray):
            raise TypeError(f"`model_out` must be type: np.ndarray but got {type(model_out)}.")

        if model_out.ndim > 2:
            raise ValueError(f"`model_out` dimension must be <= 2 but got {model_out.ndim}.")

        if model_out.shape[-1] != len(categories):
            raise ValueError(
                f"The number of classes in `model_out` should be {len(categories)} "
                f"but got {model_out.shape[-1]}."
            )

        if not np.all(np.isin(model_out, [0, 1])):
            raise ValueError("`model_out` contains elements other than 0 and 1")

    def model_to_aicoco(
        self,
        aicoco_ref: Dict[str, Any],
        model_out: np.ndarray,
        dataframes: Sequence[pd.DataFrame],
        records_output_dir: Optional[str] = None,
    ) -> AiCOCOTabularOutputDataModel:
        """Write classification records as JSONL and return a file reference."""
        categories = aicoco_ref["categories"]

        def records():
            output_index = 0
            for table, row_indexes in self.iter_record_windows(
                dataframes, aicoco_ref["tables"], aicoco_ref["meta"]
            ):
                if output_index >= len(model_out):
                    raise ValueError(
                        "The number of records is not matched with the inference model output."
                    )
                cls_pred = np.atleast_1d(model_out[output_index])
                output_index += 1
                yield {
                    "id": generate(),
                    "table_id": table.id,
                    "row_indexes": row_indexes,
                    "category_ids": [
                        category.id for pred, category in zip(cls_pred, categories) if pred
                    ],
                    "regressions": None,
                }

            if output_index != len(model_out):
                raise ValueError(
                    "The number of records is not matched with the inference model output."
                )

        records_file = self.write_records_file(records(), records_output_dir=records_output_dir)
        return AiCOCOTabularOutputDataModel.model_validate(
            {**aicoco_ref, "records_file": records_file}
        )


class AiCOCOTabularRegressionOutputStrategy(BaseAiCOCOTabularOutputStrategy):
    def __call__(
        self,
        model_out: np.ndarray,
        tables: Sequence[Dict[str, Any]],
        dataframes: Sequence[pd.DataFrame],
        regressions: Sequence[Dict[str, Any]],
        meta: Dict[str, Any],
        **kwargs,
    ) -> AiCOCOTabularOutputDataModel:
        """
        Apply the AiCOCO output strategy to reformat the model's output.

        Args:
            model_out (np.ndarray): Inference model output.
            tables (Sequence[Dict[str, Any]]): List of AiCOCO table compatible dict.
            dataframes (Sequence[pd.DataFrame]): Sequence of dataframes correspond each tables.
            regressions (Sequence[Dict[str, Any]]): List of unprocessed regressions.
            meta (Dict[str, Any]): Additional metadata.

        Returns:
            AiCOCOOut: Result in AiCOCO compatible format.
        """
        self.validate_model_output(model_out)
        aicoco_ref = self.prepare_aicoco(
            tables=tables, meta=meta, dataframes=dataframes, regressions=regressions
        )
        aicoco_out = self.model_to_aicoco(
            aicoco_ref,
            model_out,
            dataframes=dataframes,
            records_output_dir=kwargs.get("records_output_dir"),
        )
        return aicoco_out

    def validate_model_output(self, model_out: np.ndarray):
        """
        Validate inference model output.
        """
        if not isinstance(model_out, np.ndarray):
            raise TypeError(f"`model_out` must be type: np.ndarray but got {type(model_out)}.")

        if model_out.ndim > 2:
            raise ValueError(f"`model_out` dimension must be <= 2 but got {model_out.ndim}.")

        if np.isinf(model_out).any():
            raise ValueError("`model_out` contains infinite or NaN values.")

    def model_to_aicoco(
        self,
        aicoco_ref: Dict[str, Any],
        model_out: np.ndarray,
        dataframes: Sequence[pd.DataFrame],
        records_output_dir: Optional[str] = None,
    ) -> AiCOCOTabularOutputDataModel:
        """Write regression records as JSONL and return a file reference."""
        regressions = aicoco_ref["regressions"]

        def records():
            output_index = 0
            for table, row_indexes in self.iter_record_windows(
                dataframes, aicoco_ref["tables"], aicoco_ref["meta"]
            ):
                if output_index >= len(model_out):
                    raise ValueError(
                        "The number of records is not matched with the inference model output."
                    )
                pred = np.atleast_1d(model_out[output_index])
                output_index += 1
                yield {
                    "id": generate(),
                    "table_id": table.id,
                    "row_indexes": row_indexes,
                    "category_ids": None,
                    "regressions": [
                        {"regression_id": reg.id, "value": float(value)}
                        for reg, value in zip(regressions, pred)
                    ],
                }

            if output_index != len(model_out):
                raise ValueError(
                    "The number of records is not matched with the inference model output."
                )

        records_file = self.write_records_file(records(), records_output_dir=records_output_dir)
        return AiCOCOTabularOutputDataModel.model_validate(
            {**aicoco_ref, "records_file": records_file}
        )


class BaseAiCOCOHybridOutputStrategy(BaseAiCOCOOutputStrategy):
    def prepare_aicoco(
        self,
        images: List[AiImage],
        categories: Optional[Sequence[Dict[str, Any]]] = None,
        regressions: Optional[Sequence[Dict[str, Any]]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> AiCOCORef:
        meta = (
            meta
            if meta is not None
            else {"category_ids": None, "regressions": None, "table_ids": None}
        )
        return super().prepare_aicoco(images, categories, regressions, meta)


class AiCOCOHybridClassificationOutputStrategy(
    BaseAiCOCOHybridOutputStrategy, AiCOCOClassificationOutputStrategy
):
    def __call__(
        self,
        model_out: np.ndarray,
        images: List[AiImage],
        tables: Sequence[Dict[str, Any]],
        categories: Sequence[Dict[str, Any]],
        **kwargs,
    ) -> AiCOCOHybridOutputDataModel:
        self.validate_model_output(model_out, categories)
        aicoco_ref = self.prepare_aicoco(images=images, categories=categories)
        aicoco_out = self.model_to_aicoco(aicoco_ref, model_out, tables)
        return aicoco_out

    def model_to_aicoco(
        self, aicoco_ref: AiCOCORef, model_out: np.ndarray, tables: Sequence[Dict[str, Any]]
    ) -> AiCOCOHybridOutputDataModel:
        updated_images, updated_meta = self.update_images_meta(
            model_out, aicoco_ref.images, aicoco_ref.meta, aicoco_ref.categories
        )
        annot_obj = {"annotations": [], "objects": []}
        aicoco_out = {
            "images": updated_images,
            "categories": aicoco_ref.categories,
            "regressions": aicoco_ref.regressions,
            "meta": updated_meta,
            "tables": tables,
            **annot_obj,
        }
        return AiCOCOHybridOutputDataModel.model_validate(aicoco_out)


class AiCOCOHybridRegressionOutputStrategy(
    BaseAiCOCOHybridOutputStrategy, AiCOCORegressionOutputStrategy
):
    def __call__(
        self,
        model_out: np.ndarray,
        images: List[AiImage],
        tables: Sequence[Dict[str, Any]],
        regressions: Sequence[Dict[str, Any]],
        **kwargs,
    ) -> AiCOCOHybridOutputDataModel:
        self.validate_model_output(model_out)
        aicoco_ref = self.prepare_aicoco(images=images, regressions=regressions)
        aicoco_out = self.model_to_aicoco(aicoco_ref, model_out, tables)
        return aicoco_out

    def model_to_aicoco(
        self, aicoco_ref: AiCOCORef, model_out: np.ndarray, tables: Sequence[Dict[str, Any]]
    ) -> AiCOCOHybridOutputDataModel:
        updated_images, updated_meta = self.update_images_meta(
            model_out, aicoco_ref.images, aicoco_ref.meta, aicoco_ref.regressions
        )
        annot_obj = {"annotations": [], "objects": []}
        aicoco_out = {
            "images": updated_images,
            "categories": aicoco_ref.categories,
            "regressions": aicoco_ref.regressions,
            "meta": updated_meta,
            "tables": tables,
            **annot_obj,
        }
        return AiCOCOHybridOutputDataModel.model_validate(aicoco_out)
