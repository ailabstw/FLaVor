import logging
from abc import abstractmethod
from typing import Any, List, Optional, Sequence, Tuple

from pydantic import ValidationError

from ..data_models.functional import AiImage, InferCategory, InferRegression
from .base_inference_model import BaseAiCOCOInferenceModel


class BaseAiCOCOImageInferenceModel(BaseAiCOCOInferenceModel):
    """
    Base class for defining inference model with AiCOCO format response.

    This class serves as a template for implementing inference functionality for various machine learning or deep learning models.
    Subclasses must override abstract methods to define model-specific behavior.

    Attributes:
        network (Callable): The inference network or model.
    """

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
        if not images:
            raise ValueError("`images` are not provided.")

        self.images = []

        if not files:
            self.images = images
            return

        # 3D input
        if len(files) == 1 and len(images) > 1:
            self.images = images
            return

        if len(files) != len(images):
            raise ValueError(
                f"Input number of `files`({len(files)}) and `images`({len(images)}) are not matched."
            )

        for file in files:
            matched_image = next(
                (img for img in images if img["file_name"].replace("/", "_") in file), None
            )
            if matched_image:
                self.images.append(matched_image)
            else:
                raise Exception(f"Filename {file} not matched.")

        self.check_images()

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
        raise NotImplementedError

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
            if not isinstance(modified_filenames, Sequence):
                raise TypeError("`modified_filenames` should have type of `Sequence`.")

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
        A default operation for transformations which is identical transformation.

        Override it if you need other transformations like resizing or cropping, etc.

        Args:
            data (Any): Input data.

        Returns:
            Any: Preprocessed data.
        """
        return data

    def inference(self, x: Any) -> Any:
        """
        A default inference operation which performs forward operation of your defined network.

        Override it if needed.

        Args:
            x (Any): Input data.

        Returns:
            Any: Inference result.
        """

        return self.network(x)

    def postprocess(self, out: Any, metadata: Any = None) -> Any:
        """
        A default operation for post-processing which is identical transformation.

        Override it if you need activations like softmax or sigmoid generating the prediction.

        Args:
            out (Any): Inference result.
            metadata (Any, optional): Additional metadata. Defaults to None.

        Returns:
            Any: Post-processed result.
        """

        return out

    @abstractmethod
    def output_formatter(
        self,
        model_out: Any,
        images: Optional[Sequence[AiImage]] = None,
        categories: Optional[Sequence[InferCategory]] = None,
        regressions: Optional[Sequence[InferRegression]] = None,
        **kwargs,
    ) -> Any:
        """
        Abstract method to format the output of image inference model.
        To respond results in AiCOCO format, users should adopt output strategy specifying for various tasks.

        Args:
            model_out (Any): Inference output.
            images (Optional[Sequence[AiImage]], optional): List of images. Defaults to None.
            categories (Optional[Sequence[InferCategory]], optional): List of inference categories. Defaults to None.
            regressions (Optional[Sequence[InferRegression]], optional): List of inference regressions. Defaults to None.

        Returns:
            Any: AiCOCO formatted output.
        """
        raise NotImplementedError

    def check_images(self):
        """
        Check if defined images field is valid for AiCOCO format.
        """
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

    def __call__(self, **inputs: dict) -> Any:
        """
        Run the inference model.
        """
        self._set_images(**inputs)

        data, modified_filenames, metadata = self.data_reader(**inputs)

        self._update_images(modified_filenames=modified_filenames, **inputs)

        x = self.preprocess(data)
        out = self.inference(x)
        out = self.postprocess(out, metadata=metadata)

        result = self.output_formatter(
            out, images=self.images, categories=self.categories, regressions=self.regressions
        )

        return result
