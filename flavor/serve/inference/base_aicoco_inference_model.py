import logging
import warnings
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
from fastapi import UploadFile
from nanoid import generate
from pydantic import BaseModel, ValidationError

from flavor.serve.models import (
    AiAnnotation,
    AiCategory,
    AiImage,
    AiMeta,
    AiObject,
    AiRegression,
    InferCategory,
    InferRegression,
)

from .base_inference_model import BaseInferenceModel


class BaseAiCOCOInputDataModel(BaseModel):
    """
    Base class for defining input data model with AiCOCO format.

    Inherit it if you need extra fields.

    Attributes:
        images (Optional[Sequence[AiImage]]): Sequence of AiImage objects. Defaults to None.
        files (Optional[Sequence[UploadFile]]): Sequence of UploadFile objects. Defaults to None.

    Example:
    ```
    class InputDataModel(BaseAiCOCOInputDataModel):
        image_embeddings: NpArray

    InputDataModel(
        {
            "images": ...,
            "image_embeddings": ...
        }
    )
    ```
    """

    images: Optional[Sequence[AiImage]] = None
    files: Optional[Sequence[UploadFile]] = None


class BaseAiCOCOOutputDataModel(BaseModel):
    """
    Base class for defining output data model with AiCOCO format.

    Inherit it if you need extra fields.

    Attributes:
        images (Optional[Sequence[AiImage]]): Sequence of AiImage objects. Defaults to None.
        annotations (Optional[Sequence[AiAnnotation]]): Sequence of AiAnnotation objects. Defaults to None.
        categories (Optional[Sequence[AiCategory]]): Sequence of AiCategory objects. Defaults to None.
        regressions (Optional[Sequence[AiRegression]]): Sequence of AiRegression objects. Defaults to None.
        objects (Optional[Sequence[AiObject]]): Sequence of AiObject objects. Defaults to None.
        meta (Optional[AiMeta]): AiMeta object. Defaults to None.

    Example:
    ```
    class OutputDataModel(BaseAiCOCOOutputDataModel):
        mask_bin: NpArray

    OutputDataModel(
        {
            "images": ...,
            "annotations": ...,
            "categories": ...,
            "objects": ...,
            "meta": ...,
            "mask_bin": ...
        }
    )
    ```
    """

    images: Optional[Sequence[AiImage]] = None
    annotations: Optional[Sequence[AiAnnotation]] = None
    categories: Optional[Sequence[AiCategory]] = None
    regressions: Optional[Sequence[AiRegression]] = None
    objects: Optional[Sequence[AiObject]] = None
    meta: Optional[AiMeta] = None


class BaseAiCOCOInferenceModel(BaseInferenceModel):
    """
    Base class for defining inference model with AiCOCO format response.

    This class serves as a template for implementing inference functionality for various machine learning or deep learning models.
    Subclasses must override abstract methods to define model-specific behavior.

    Attributes:
        network (Callable): The inference network or model.
    """

    def __init__(self):
        self.network = self.define_inference_network()

    @abstractmethod
    def define_inference_network(self) -> Callable:
        """
        Abstract method to define the inference network.

        Returns:
            Callable: The defined inference network instance.
                The return value would be assigned to `self.network`.
        """
        pass

    @abstractmethod
    def set_categories(self) -> Optional[List[InferCategory]]:
        """
        Abstract method to set inference categories. Return `None` if no categories.

        Returns:
            List[InferCategory]: A list defining inference categories.
        """
        pass

    @abstractmethod
    def set_regressions(self) -> Optional[List[InferRegression]]:
        """
        Abstract method to set inference regressions. Return `None` if no regressions.

        Returns:
            List[InferRegression]: A list defining inference regressions.
        """
        pass

    def _set_images(self, files: Sequence[str] = [], images: Sequence[AiImage] = [], **kwargs):
        """
        Set `self.images` attribute for AiCOCO format.
        This method takes `files` and `images` as inputs and we expect at least one of them should not be empty list.
        It is not recommended to have both values empty list as it would result in an empty list for `self.image`.
        Users must be aware of this behavior and handle `self.images` afterwards if AiCOCO format is desired.

        Args:
            files (Sequence[str]): List of input filenames. Defaults to [].
            images (Sequence[AiImage], optional): List of AiCOCO image elements. Defaults to [].

        """
        self.images = []
        if files and images:
            if len(files) == len(images):
                # check if `files` in formdata is matched with `file_name` in AiCOCO input
                for file in files:
                    try:
                        self.images.append(
                            next(
                                format_image
                                for format_image in images
                                if format_image["file_name"].replace("/", "_") in file
                            )
                        )
                    except StopIteration:
                        raise Exception(f"Filename {file} not matched.")
            elif len(files) == 1 and len(images) > 1:
                self.images = images
            else:
                raise ValueError(
                    f"Input number of `files`({len(files)}) and `images`({len(images)}) not supported."
                )
        elif files and not images:
            # Not sure if we should handle this
            for file in files:
                self.images.append(
                    {
                        "id": generate(),
                        "index": 0,
                        "file_name": file,
                        "category_ids": None,
                        "regressions": None,
                    }
                )
        elif not files and images:
            self.images = images
        else:
            warnings.warn("`files` and `images` not specified.")

    @abstractmethod
    def data_reader(
        self, files: Sequence[str], **kwargs
    ) -> Tuple[Any, Optional[List[str]], Optional[Any]]:
        """
        Abstract method to read data for inference model.
        This method should return three things:
        1. data: in np.ndarray or torch.Tensor for inference model.
        2. modified_filenames: modified list of filenames if the order of `files` is altered.
        3. metadata: useful metadata for the following steps.

        Args:
            files (Sequence[str]): List of input filenames.

        Returns:
            Tuple[Any, Optional[List[str]], Optional[Any]]: A tuple containing data, modified filenames, and metadata.
        """
        pass

    def _update_images(
        self, files: Sequence[str] = [], modified_filenames: Sequence[str] = None, **kwargs
    ):
        """
        Update the images based on modified filenames.

        Args:
            files (Sequence[str]): List of input filenames.
            modified_filenames (Sequence[str]): List of modified filenames.
        """
        if modified_filenames is not None and files:
            if len(self.images) != len(modified_filenames):
                raise ValueError(
                    "`self.images` and `modified_filenames` have different amount of elements."
                )
            updated_indices = []
            for file in modified_filenames:
                updated_indices.append(files.index(file))
            if len(self.images) != len(updated_indices):
                raise ValueError(
                    "Each element of `images` and `modified_filenames` should be matched."
                )
            self.images = [self.images[i] for i in updated_indices]

    def preprocess(self, data: Any) -> Any:
        """
        Preprocess the input data where transformations like resizing and cropping operated.

        Override it if needed.

        Args:
            data (Any): Input data.

        Returns:
            Any: Preprocessed data.
        """
        return data

    def inference(self, x: Any) -> Any:
        """
        Perform inference.

        Override it if needed.

        Args:
            x (Any): Input data.

        Returns:
            Any: Inference result.
        """

        return self.network(x)

    def postprocess(self, out: Any, metadata: Any = None) -> Any:
        """
        Postprocess the inference result where activations like softmax or sigmoid operated.

        Override it if needed.

        Args:
            out (Any): Inference result.
            metadata (Any, optional): Additional metadata. Defaults to None.

        Returns:
            Any: Postprocessed result.
        """

        return out

    @abstractmethod
    def output_formatter(
        self,
        model_out: Any,
        images: Optional[Sequence[AiImage]] = None,
        categories: Optional[List[InferCategory]] = None,
        regressions: Optional[List[InferRegression]] = None,
        **kwargs,
    ) -> Any:
        """
        Abstract method to format the output of inference model.
        To respond results in AiCOCO format, users should adopt output strategy specifying for various tasks.

        Args:
            model_out (Any): Inference output.
            images (Optional[Sequence[AiImage]], optional): List of images. Defaults to None.
            categories (Optional[List[InferCategory]], optional): List of inference categories. Defaults to None.
            regressions (Optional[List[InferRegression]], optional): List of inference regressions. Defaults to None.

        Returns:
            Any: Formatted output.
        """
        pass

    def __call__(self, **net_input) -> Any:
        """
        Run the inference model.
        """
        self.categories = self.set_categories()
        if self.categories is not None:
            if isinstance(self.categories, Sequence):
                try:
                    for c in self.categories:
                        InferCategory.model_validate(c)
                except ValidationError:
                    logging.error(
                        "Each element of `categories` should have format of `InferCategory`."
                    )
                    raise
            else:
                raise TypeError("`categories` should have type of `Sequence[InferCategory]`.")

        self.regressions = self.set_regressions()
        if self.regressions is not None:
            if isinstance(self.regressions, Sequence):
                try:
                    for r in self.regressions:
                        InferRegression.model_validate(r)
                except ValidationError:
                    logging.error(
                        "Each element of `regressions` should have format of `InferRegression`."
                    )
                    raise
            else:
                raise TypeError("`regressions` should have type of `Sequence[InferRegression]`.")

        self._set_images(**net_input)
        if self.images is None:
            raise ValueError("`self.images` should not be None.")
        else:
            if isinstance(self.images, Sequence):
                try:
                    for i in self.images:
                        AiImage.model_validate(i)
                except ValidationError:
                    logging.error("Each element of `self.images` should have format of `AiImage`.")
                    raise
            else:
                raise TypeError("`self.images` should have type of `Sequence[AiImage]`.")

        data, modified_filenames, metadata = self.data_reader(**net_input)
        if modified_filenames is not None:
            if not isinstance(modified_filenames, Sequence):
                raise TypeError("`modified_filenames` should have type of `Sequence`.")

        self._update_images(modified_filenames=modified_filenames, **net_input)
        with torch.no_grad():
            x = self.preprocess(data)
            out = self.inference(x)
            out = self.postprocess(out, metadata=metadata)
        result = self.output_formatter(
            out, images=self.images, categories=self.categories, regressions=self.regressions
        )

        return result
