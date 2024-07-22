from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel

from ..data_models.functional import AiImage
from .base_inference_model import BaseInferenceModel


class InferCategory(BaseModel):
    name: str
    supercategory_name: Optional[str] = None


class InferRegression(BaseModel):
    name: str
    superregression_name: Optional[str] = None


class BaseAiCOCOInferenceModel(BaseInferenceModel):
    def __init__(self):
        self.network = self.define_inference_network()
        self.categories = self.set_categories()
        self.check_categories()

        self.regressions = self.set_regressions()
        self.check_regressions()

    def check_categories(self):
        """
        Check if defined categories field is valid for AiCOCO format.
        """
        if self.categories is not None:
            if isinstance(self.categories, Sequence):
                if not all(InferCategory.model_validate(c) for c in self.categories):
                    raise ValueError(
                        "Not all elements in `self.categories` is valid for category structure."
                    )
            else:
                raise TypeError("`categories` should have type of `Sequence[InferCategory]`.")

    def check_regressions(self):
        """
        Check if defined regressions field is valid for AiCOCO format.
        """
        if self.regressions is not None:
            if isinstance(self.regressions, Sequence):
                if not all(InferRegression.model_validate(c) for c in self.regressions):
                    raise ValueError(
                        "Not all elements in `self.regressions` is valid for regression structure."
                    )
            else:
                raise TypeError("`regressions` should have type of `Sequence[InferRegression]`.")

    @abstractmethod
    def define_inference_network(self) -> Callable:
        """
        Abstract method to define the inference network.

        Returns:
            Callable: The defined inference network instance.
                The return value would be assigned to `self.network`.
        """
        raise NotImplementedError

    @abstractmethod
    def set_categories(self) -> Optional[List[Dict[str, Any]]]:
        """
        Abstract method to set inference categories. Return `None` if no categories.

        Returns:
            List[Dict[str, Any]]: A list defining inference categories.
        """
        raise NotImplementedError

    @abstractmethod
    def set_regressions(self) -> Optional[List[Dict[str, Any]]]:
        """
        Abstract method to set inference regressions. Return `None` if no regressions.

        Returns:
            List[Dict[str, Any]]: A list defining inference regressions.
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, net_input: Any) -> Any:
        """
        Abstract method to preprocess the input data where transformations like resizing and cropping operated.

        Args:
            data (Any): Input data.
        """
        raise NotImplementedError

    @abstractmethod
    def inference(self, x: Any) -> Any:
        """
        Abstract method to perform inference.

        Override it if needed.

        Args:
            x (Any): Input data.
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, out: Any) -> Any:
        """
        Abstract method to post-process the inference result where activations like softmax or sigmoid performed.

        Args:
            out (Any): Inference result.
        """
        raise NotImplementedError

    @abstractmethod
    def output_formatter(
        self,
        model_out: Any,
        categories: Optional[Sequence[Dict[str, Any]]] = None,
        regressions: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """
        Abstract method to format the output of inference model.
        This is just a template for you to make sure you make use of `categories` and `regressions`.
        Override it with your additional arguments such as `images`.

        Args:
            model_out (Any): Inference output.
            categories (Optional[Sequence[Dict[str, Any]]]): List of inference categories. Default: None.
            regressions (Optional[Sequence[Dict[str, Any]]]): List of inference regressions. Default: None.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, **inputs) -> Any:
        """
        Abstract method to run inference model.

        This method orchestrates the entire inference process: preprocessing,
        inference, postprocessing, and output formatting.

        For example:

        ```
        x = self.preprocess(**inputs)
        out = self.inference(x)
        out = self.postprocess(out)

        # generate result in specific format
        result = self.output_formatter(
            out, categories=self.categories, regressions=self.regressions, **inputs
        )

        return result
        ```
        """
        raise NotImplementedError


class BaseAiCOCOImageInferenceModel(BaseAiCOCOInferenceModel):
    """
    Base class for defining inference model with AiCOCO format response.

    This class serves as a template for implementing inference functionality for various machine learning or deep learning models.
    Subclasses must override abstract methods to define model-specific behavior.

    Attributes:
        network (Callable): The inference network or model.
    """

    @abstractmethod
    def define_inference_network(self) -> Callable:
        """
        Abstract method to define the inference network.

        Returns:
            Callable: The defined inference network instance.
                The return value would be assigned to `self.network`.
        """
        raise NotImplementedError

    @abstractmethod
    def set_categories(self) -> Optional[List[Dict[str, Any]]]:
        """
        Abstract method to set inference categories. Return `None` if no categories.

        Returns:
            List[Dict[str, Any]]: A list defining inference categories.
        """
        raise NotImplementedError

    @abstractmethod
    def set_regressions(self) -> Optional[List[Dict[str, Any]]]:
        """
        Abstract method to set inference regressions. Return `None` if no regressions.

        Returns:
            List[Dict[str, Any]]: A list defining inference regressions.
        """
        raise NotImplementedError

    def _set_images(
        self, images: Sequence[Dict[str, Any]], files: Optional[Sequence[str]] = None
    ) -> List[AiImage]:
        """
        Initialize `self.images` attribute for AiCOCO format.
        This method takes `files` and `images` as inputs and we expect `images` is always provided.
        It is not recommended to have both `files` and `images` empty as it would result in an empty list for `self.image`.
        Users must be aware of this behavior and handle `self.images` afterwards if AiCOCO format is desired.

        This method perform following steps if `files` exists:
            1. len(files) == len(set(files))
            2. set(files) == set(images_file_name)
            3. sort by index
            4. sort by files

        Args:
            images (Sequence[AiImage]): List of AiCOCO image elements.
            files (Optional[Sequence[str]]): List of input filenames. Default: None.
        """
        self.images = []

        # set `self.images` by the order of its index attribute.
        if not files:
            for sorted_img in sorted(images, key=lambda x: x["index"]):
                self.images.append(AiImage.model_validate(sorted_img))
            return

        assert len(files) == len(set(files)), "Each element of `files` should be unique."
        m = len(set(files))
        n = len(set([img["file_name"] for img in images]))
        assert (
            m == n
        ), f"The number of `files` and `file_name` in `images` is not valid. `files` has {m} unique elements but file_name in `images` has {n} unique elements."

        sorted_images = sorted(images, key=lambda x: x["index"])

        for target_file in files:
            # Ideally, `target_file` would be `image` with some hash prefix.
            for image in sorted_images:
                image_file_name = image["file_name"].replace("/", "_")
                if target_file.endswith(image_file_name):
                    self.images.append(AiImage.model_validate(image))
                    break  # early break if found matched
            else:
                raise ValueError(f"{target_file} could not be found in input `images`.")

    @abstractmethod
    def data_reader(
        self, files: Optional[Sequence[str]] = None, **kwargs
    ) -> Tuple[Any, Optional[List[str]], Optional[Any]]:
        """
        Abstract method to read data for inference model.
        This method should return three things:
        1. data: in np.ndarray or torch.Tensor for inference model.
        2. modified_filenames: modified list of filenames if the order of `files` is altered (e.g., 3D multiple slices input).
        3. metadata: necessary metadata for the post-processing.

        Args:
            files (Sequence[str]): List of input filenames.

        Returns:
            Tuple[Any, Optional[List[str]], Optional[Any]]: A tuple containing data, modified filenames, and metadata.
        """
        raise NotImplementedError

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

    def postprocess(self, out: Any, metadata: Optional[Any] = None) -> Any:
        """
        A default operation for post-processing which is identical transformation.

        Override it if you need activations like softmax or sigmoid generating the prediction.

        Args:
            out (Any): Inference result.
            metadata (Any, optional): Additional metadata. Default: None.

        Returns:
            Any: Post-processed result.
        """

        return out

    @abstractmethod
    def output_formatter(
        self,
        model_out: Any,
        categories: Optional[Sequence[Dict[str, Any]]] = None,
        regressions: Optional[Sequence[Dict[str, Any]]] = None,
        images: Optional[Sequence[AiImage]] = None,
        **kwargs,
    ) -> Any:
        """
        Abstract method to format the output of image inference model.
        To respond results in AiCOCO format, users should adopt output strategy specifying for various tasks.

        Args:
            model_out (Any): Inference output.
            categories (Optional[Sequence[Dict[str, Any]]]): List of inference categories. Default: None.
            regressions (Optional[Sequence[Dict[str, Any]]]): List of inference regressions. Default: None.
            images (Optional[Sequence[AiImage]]): List of images. Default: None.

        Returns:
            Any: AiCOCO formatted output.
        """
        raise NotImplementedError

    def __call__(self, images: Dict[str, Any], files: Optional[str] = None, **kwargs) -> Any:
        """
        Run the inference model.
        """
        data, modified_files, metadata = self.data_reader(files=files, **kwargs)
        self._set_images(images=images, files=modified_files if modified_files else files)

        x = self.preprocess(data)
        out = self.inference(x)
        out = self.postprocess(out, metadata=metadata)

        result = self.output_formatter(
            out,
            categories=self.categories,
            regressions=self.regressions,
            images=self.images,
        )

        return result
