import copy
from abc import abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2  # type: ignore
import numpy as np
from nanoid import generate  # type: ignore

from ..models import (
    AiAnnotation,
    AiCategory,
    AiCOCOFormat,
    AiImage,
    AiMeta,
    AiObject,
    AiRegression,
    InferCategory,
    InferClassificationOutput,
    InferDetectionOutput,
    InferRegression,
    InferRegressionOutput,
    InferSegmentationOutput,
    ModelOut,
)
from .base_strategy import BaseStrategy

AiCOCOOut = Dict[
    str, Union[Sequence[AiImage], Sequence[AiCategory], Sequence[AiRegression], AiMeta]
]


class BaseAiCOCOOutputStrategy(BaseStrategy):
    @abstractmethod
    def model_to_aicoco(self, aicoco_out: AiCOCOOut, model_out: ModelOut, **kwargs) -> AiCOCOFormat:
        """
        Abstract method to convert model output to AiCOCO compatible format.
        """
        raise NotImplementedError

    def generate_images(self, images: Sequence[AiImage]) -> List[AiImage]:
        """
        Generate `images` field in AiCOCO compatible format by removing physical_file_name.

        Args:
            images (Sequence[AiImage]): Modified AiCOCO `images` field.

        Returns:
            List[AiImage]: AiCOCO `images` field.
        """
        for image in images:
            if "physical_file_name" in image:
                image.pop("physical_file_name", None)

        return images

    def generate_categories(self, categories: Sequence[InferCategory]) -> List[InferCategory]:
        """
        Generate `categories` field in AiCOCO compatible format.

        Args:
            categories (Sequence[InferCategory]): Dictionary mapping class indices to category nanoid.

        Returns:
            List[InferCategory]: AiCOCO `categories` field.
        """
        res = list()
        supercategory_id_table = dict()

        for class_id, category in enumerate(categories):
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

    def generate_regressions(self, regressions: Sequence[InferRegression]) -> List[InferRegression]:
        """
        Generate `regressions` field in AiCOCO compatible format.

        Args:
            regressions (Sequence[InferRegression]): Dictionary mapping regression indices to regression nanoid.

        Returns:
            List[InferRegression]: AiCOCO `regressions` field.
        """
        res = list()
        superregression_id_table = dict()

        for regression_id, regression in enumerate(regressions):
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
        images: Sequence[AiImage],
    ) -> None:
        """
        Set table attributes for mapping images, class, and regression IDs.
        The class and regression ID tables are only set in the first inference run.

        Args:
            images (Sequence[AiImage]): AiCOCO `images` field.
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
        model_out: ModelOut,
        images: Sequence[AiImage],
        categories: Optional[Sequence[InferCategory]] = [],
        regressions: Optional[Sequence[InferRegression]] = [],
        meta: AiMeta = {"category_ids": None, "regressions": None},
    ) -> Tuple[AiCOCOOut, ModelOut]:
        """
        Prepare prerequisite for AiCOCO.

        Args:
            model_out (ModelOut): Inference model output.
            images (Sequence[AiImage]): List of AiCOCO images field.
            meta (AiMeta): AiCOCO meta field. Default: {"category_ids": None, "regressions": None}.
            categories (Sequence[InferCategory]): List of unprocessed categories. Default: [].
            regressions (Sequence[InferRegression]): List of unprocessed regressions. Default: [].
        Returns:
            Tuple[AiCOCOOut, ModelOut]: Prepared AiCOCO output and inference model output array.
        """
        if categories is None:
            categories = {}
        if regressions is None:
            regressions = {}

        if not hasattr(self, "aicoco_categories") and not hasattr(self, "aicoco_regressions"):
            # only activate at first run
            self.aicoco_categories = self.generate_categories(copy.deepcopy(categories))
            self.aicoco_regressions = self.generate_regressions(copy.deepcopy(regressions))
            self.set_images_class_regression_id_table(images)

        aicoco_out = {
            "images": self.generate_images(copy.deepcopy(images)),
            "categories": self.aicoco_categories,
            "regressions": self.aicoco_regressions,
            "meta": copy.deepcopy(meta),
        }

        return aicoco_out, model_out


class AiCOCOSegmentationOutputStrategy(BaseAiCOCOOutputStrategy):
    def __call__(
        self, images: Sequence[AiImage], categories: Sequence[InferCategory], model_out: np.ndarray
    ) -> AiCOCOFormat:
        """
        Apply the AiCOCO output strategy to reformat the model's output.

        Args:
            images (Sequence[AiImage]): List of AiCOCO images field.
            categories (Sequence[InferCategory]): List of unprocessed categories.
            model_out (np.ndarray): Inference model output.

        Returns:
            AiCOCOFormat: Result in complete AiCOCO format.
        """
        aicoco_out, model_out = self.prepare_aicoco(
            images=images, categories=categories, model_out=model_out
        )
        response = self.model_to_aicoco(aicoco_out, model_out)
        AiCOCOFormat.model_validate(response)
        return response

    def prepare_aicoco(
        self, **infer_output: InferSegmentationOutput
    ) -> Tuple[AiCOCOOut, np.ndarray]:
        """Validate and prepare inputs for AiCOCO Segmentation Output Strategy.

        Returns:
            Tuple[AiCOCOOut, np.ndarray]: Prepared AiCOCO output and inference model output array.
        """
        InferSegmentationOutput.model_validate(infer_output)
        return super().prepare_aicoco(**infer_output)

    def model_to_aicoco(self, aicoco_out: AiCOCOOut, model_out: ModelOut) -> AiCOCOFormat:
        """
        Convert segmentation inference model output to AiCOCO compatible format.

        Args:
            aicoco_out (AiCOCOOut): Complete AiCOCO output.
            model_out (ModelOut): Segmentation inference model output.

        Returns:
            AiCOCOFormat: Result in AiCOCO compatible format.
        """
        annot_obj = self.generate_annotations_objects(model_out)

        return {**aicoco_out, **annot_obj}

    def generate_annotations_objects(
        self, out: ModelOut
    ) -> Dict[str, Union[Sequence[AiAnnotation], Sequence[AiObject]]]:
        """
        Generate `annotations` and `objects` in AiCOCO compatible format from 4D volumetric data.

        Args:
            out (ModelOut): 3D or 4D predicted seg inference model output. This could be regular semantic seg mask or grouped instance seg mask.
        Returns:
            Dict[str, Union[Sequence[AiAnnotation], Sequence[AiObject]]]: Dictionary containing annotations and objects in AiCOCO compatible format.

        Notes:
            - The function assumes the input data is preprocessed with connected regions labeled with indices if it is an instance segmentation task.
            - Annotations are generated for each labeled region in each slice.
            - 'bbox' in annotations is set to None, and 'segmentation' is defined based on contours.
            - The 'iscrowd' field in annotations is set to 0 for non-crowd objects.
        """

        assert (
            out.ndim == 3 or out.ndim == 4
        ), f"dim of the inference model output {out.shape} should be in 3D or 4D."

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


class AiCOCOClassificationOutputStrategy(BaseAiCOCOOutputStrategy):
    def __call__(
        self, images: Sequence[AiImage], categories: Sequence[InferCategory], model_out: np.ndarray
    ) -> AiCOCOFormat:
        """
        Apply the AiCOCO output strategy to reformat the model's output.

        Args:
            images (Sequence[AiImage]): List of AiCOCO images field.
            categories (Sequence[InferCategory]): List of unprocessed categories.
            model_out (np.ndarray): Inference model output.

        Returns:
            AiCOCOFormat: Result in complete AiCOCO format.
        """
        aicoco_out, model_out = self.prepare_aicoco(
            images=images, categories=categories, model_out=model_out
        )
        response = self.model_to_aicoco(aicoco_out, model_out)
        AiCOCOFormat.model_validate(response)
        return response

    def prepare_aicoco(
        self, **infer_output: InferClassificationOutput
    ) -> Tuple[AiCOCOOut, np.ndarray]:
        """Validate and prepare inputs for AiCOCO Classification Output Strategy.

        Returns:
            Tuple[AiCOCOOut, np.ndarray]: Prepared AiCOCO output and inference model output array.
        """
        InferClassificationOutput.model_validate(infer_output)
        return super().prepare_aicoco(**infer_output)

    def model_to_aicoco(
        self,
        aicoco_out: AiCOCOOut,
        model_out: ModelOut,
    ) -> AiCOCOFormat:
        """
        Convert classification inference model output to AiCOCO compatible format.

        Args:
            aicoco_out (AiCOCOOut): Complete AiCOCO output.
            model_out (ModelOut): Classification inference model output.

        Returns:
            AiCOCOFormat: Result in AiCOCO compatible format.
        """
        aicoco_out["images"], aicoco_out["meta"] = self.update_images_meta(
            model_out, aicoco_out["images"], aicoco_out["meta"]
        )

        annot_obj = {"annotations": [], "objects": []}

        return {**aicoco_out, **annot_obj}

    def update_images_meta(
        self,
        out: ModelOut,
        images: Sequence[AiImage],
        meta: AiMeta,
    ) -> Tuple[List[AiImage], AiMeta]:
        """
        Update `category_ids` in  `images` and `meta` based on the classification model output.

        Args:
            out (ModelOut): Inference model output in 1D shape: (n,).
            images (Sequence[AiImage]): List of AiCOCO images field.
            meta (AiMeta): AiCOCO meta field.

        Returns:
            Tuple[List[AiImage], AiMeta]: Updated AiCOCO compatible images and meta field.

        Notes:
            The function updates the 'category_ids' field in both the images and meta dictionaries based on the inference model output.
            For 2D input data, it updates the last element of the images field list.
            For 3D input data, it updates meta field.
        """
        assert out.ndim == 1, f"dim of the inference model output {out.shape} should be 1D."
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


class AiCOCODetectionOutputStrategy(BaseAiCOCOOutputStrategy):
    def __call__(
        self,
        images: Sequence[AiImage],
        categories: Sequence[InferCategory],
        regressions: Sequence[InferRegression],
        model_out: np.ndarray,
    ) -> AiCOCOFormat:
        """
        Apply the AiCOCO output strategy to reformat the model's output.

        Args:
            images (Sequence[AiImage]): List of AiCOCO images field.
            categories (Sequence[InferCategory]): List of unprocessed categories.
            regressions (Sequence[InferRegression]): List of unprocessed regressions.
            model_out (np.ndarray): Inference model output.

        Returns:
            AiCOCOFormat: Result in complete AiCOCO format.
        """
        aicoco_out, model_out = self.prepare_aicoco(
            images=images, categories=categories, regressions=regressions, model_out=model_out
        )
        response = self.model_to_aicoco(aicoco_out, model_out)
        AiCOCOFormat.model_validate(response)
        return response

    def prepare_aicoco(self, **infer_output: InferDetectionOutput) -> Tuple[AiCOCOOut, np.ndarray]:
        """Validate and prepare inputs for AiCOCO Detection Output Strategy.

        Returns:
            Tuple[AiCOCOOut, np.ndarray]: Prepared AiCOCO output and inference model output array.
        """
        InferDetectionOutput.model_validate(infer_output)
        return super().prepare_aicoco(**infer_output)

    def model_to_aicoco(
        self,
        aicoco_out: AiCOCOOut,
        model_out: ModelOut,
    ) -> AiCOCOFormat:
        """
        Convert detection inference model output to AiCOCO compatible format.

        Args:
            aicoco_out (AiCOCOOut): Complete AiCOCO output.
            model_out (ModelOut): Detection inference model output.

        Returns:
            AiCOCOFormat: Result in AiCOCO compatible format.
        """
        annot_obj = self.generate_annotations_objects(model_out)

        return {**aicoco_out, **annot_obj}

    def generate_annotations_objects(
        self,
        out: ModelOut,
    ) -> Dict[str, Union[Sequence[AiAnnotation], Sequence[AiObject]]]:
        """
        Generate `annotations` and `objects` in AiCOCO compatible format from detection model output.

        Args:
            out (ModelOut): Detection inference model output dictionary with keys:
                - 'bbox_pred': List of bounding box predictions in the format [y_min, x_min, y_max, x_max].
                - 'cls_pred': List of one-hot classification result for each bbox.
                - 'confidence_score' (optional): List of confidence scores for each prediction.
                - 'regression_value' (optional): List of regression values for each prediction.
            Please refer to `DetModelOut` for more detail.

        Returns:
            Dict[str, Union[Sequence[AiAnnotation], Sequence[AiObject]]]: Dictionary containing annotations and objects in AiCOCO compatible format.

        Notes:
            - The function generates unique object IDs, image IDs, and annotation IDs using the `generate` function.
            - If 'confidence_score' is present in the output, it is added to the generated objects.
            - If 'regressions' are present in the output, they are added to the generated objects.
            - The 'bbox' in annotations is in the format [[x_min, y_min, x_max, y_max]].
            - The 'segmentation' in annotations is set to None for detection tasks.
        """
        assert isinstance(out, dict), "The type of inference model output should be `dict`."
        assert "bbox_pred" in out, "A key `bbox_pred` must be in inference model output."
        assert "cls_pred" in out, "A key `cls_pred` must be in inference model output."

        res = dict()
        res["annotations"] = list()
        res["objects"] = list()

        for i, bbox_pred in enumerate(out["bbox_pred"]):
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

            cls_pred = out["cls_pred"][i]
            assert len(cls_pred) == len(
                self.class_id_table
            ), f"The number of categories is not matched with the detection output {cls_pred.shape}."

            for c in range(len(cls_pred)):
                if cls_pred[c] == 0 or c not in self.class_id_table:
                    continue
                obj["category_ids"].append(self.class_id_table[c])

            if not obj["category_ids"]:
                # all the `display` flag in the predicted classes are false
                continue

            confidence_score = out.get("confidence_score")
            if confidence_score:
                cs = confidence_score[i]
                obj["confidence"] = cs.item() if isinstance(cs, np.ndarray) else cs

            regression_value = out.get("regression_value")
            if regression_value:
                obj["regressions"] = list()
                for i, value in enumerate(regression_value[i]):
                    obj["regressions"].append(
                        {
                            "regression_id": self.regression_id_table[i],
                            "value": value.item() if isinstance(value, np.ndarray) else value,
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


class AiCOCORegressionOutputStrategy(BaseAiCOCOOutputStrategy):
    def __call__(
        self,
        images: Sequence[AiImage],
        regressions: Sequence[InferRegression],
        model_out: np.ndarray,
    ) -> AiCOCOFormat:
        """
        Apply the AiCOCO output strategy to reformat the model's output.

        Args:
            images (Sequence[AiImage]): List of AiCOCO images field.
            regressions (Sequence[InferRegression]): List of unprocessed regressions.
            model_out (np.ndarray): Inference model output.

        Returns:
            AiCOCOFormat: Result in complete AiCOCO format.
        """
        aicoco_out, model_out = self.prepare_aicoco(
            images=images, regressions=regressions, model_out=model_out
        )
        response = self.model_to_aicoco(aicoco_out, model_out)
        AiCOCOFormat.model_validate(response)
        return response

    def prepare_aicoco(self, **infer_output: InferRegressionOutput) -> Tuple[AiCOCOOut, np.ndarray]:
        """Validate and prepare inputs for AiCOCO Regression Output Strategy.

        Returns:
            Tuple[AiCOCOOut, np.ndarray]: Prepared AiCOCO output and inference model output array.
        """
        InferRegressionOutput.model_validate(infer_output)
        return super().prepare_aicoco(**infer_output)

    def model_to_aicoco(
        self,
        aicoco_out: AiCOCOOut,
        model_out: ModelOut,
    ) -> AiCOCOFormat:
        """
        Convert regression inference model output to AiCOCO compatible format.

        Args:
            aicoco_out (AiCOCOOut): Complete AiCOCO output.
            model_out (ModelOut): Regression inference model output.

        Returns:
            AiCOCOFormat: Result in AiCOCO compatible format.
        """
        aicoco_out["images"], aicoco_out["meta"] = self.update_images_meta(
            model_out, aicoco_out["images"], aicoco_out["meta"]
        )

        annot_obj = {"annotations": [], "objects": []}

        return {**aicoco_out, **annot_obj}

    def update_images_meta(
        self,
        out: ModelOut,
        images: Sequence[AiImage],
        meta: AiMeta,
    ) -> Tuple[List[AiImage], AiMeta]:
        """
        Update `regressions` in `images` and `meta` based on the regression model output.

        Args:
            out (ModelOut): Inference model output in 1D shape: (n,).
            images (Sequence[AiImage]): List of AiCOCO images field.
            meta (AiMeta): AiCOCO meta field.

        Returns:
            Tuple[List[AiImage], AiMeta]: Updated AiCOCO compatible images and meta field.

        Notes:
            The function updates the 'regressions' field in both the images and meta dictionaries based on the model output.
            For 2D input data, it updates the last element of the images field list.
            For 3D input data, it updates meta.
        """
        assert out.ndim == 1, f"dim of the inference model output {out.shape} should be 1D."

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
