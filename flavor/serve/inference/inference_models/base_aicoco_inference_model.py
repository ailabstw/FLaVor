from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from pydantic import BaseModel

from ..data_models.functional import AiImage, AiTable
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

    def postprocess(self, out: Any) -> Any:
        """
        A default operation for post-processing which is identical transformation.

        Override it if you need activations like softmax or sigmoid generating the prediction.

        Args:
            out (Any): Inference result.

        Returns:
            Any: Post-processed result.
        """

        return out

    @abstractmethod
    def output_formatter(self, *args, **kwargs) -> Any:
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
    def __call__(self, *args, **kwargs) -> Any:
        """
        Abstract method to run inference model.

        This method orchestrates the entire inference process: preprocessing,
        inference, postprocessing, and output formatting.

        For example:

        ```
        x = self.preprocess(**kwargs)
        out = self.inference(x)
        out = self.postprocess(out)

        # generate result in specific format
        result = self.output_formatter(
            out, categories=self.categories, regressions=self.regressions, **kwargs
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

    def __init__(self):
        super().__init__()

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
        self,
        images: Optional[Sequence[Dict[str, Any]]] = None,
        files: Optional[Sequence[str]] = None,
    ) -> None:
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
            images (Optional[Sequence[Dict[str, Any]]]): List of AiCOCO image elements. Default: None.
            files (Optional[Sequence[str]]): List of input filenames. Default: None.
        """
        self.images = []

        if not images:
            return

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
            found_matched = False
            for image in sorted_images:
                image_file_name = image["file_name"].replace("/", "@@@")
                if target_file.endswith(image_file_name):
                    found_matched = True
                    self.images.append(AiImage.model_validate(image))

            if not found_matched:
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

    @abstractmethod
    def output_formatter(
        self,
        model_out: Any,
        data: Any,
        images: Sequence[AiImage],
        categories: Optional[Sequence[Dict[str, Any]]] = None,
        regressions: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """
        Abstract method to format the output of image inference model.
        To respond results in AiCOCO format, users should adopt output strategy specifying for various tasks.

        Args:
            model_out (Any): Inference output.
            images (Optional[Sequence[Dict[str, Any]]]): List of images. Default: None.
            categories (Optional[Sequence[Dict[str, Any]]]): List of inference categories. Default: None.
            regressions (Optional[Sequence[Dict[str, Any]]]): List of inference regressions. Default: None.

        Returns:
            Any: AiCOCO formatted output.
        """
        raise NotImplementedError

    def __call__(
        self,
        images: Optional[Sequence[Dict[str, Any]]] = None,
        files: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> Any:
        """
        Run the inference model.
        """
        data, modified_files = self.data_reader(files=files, **kwargs)
        self._set_images(images=images, files=modified_files if modified_files else files)

        x = self.preprocess(data)
        out = self.inference(x)
        out = self.postprocess(out)

        result = self.output_formatter(
            out,
            data=data,
            images=self.images,
            categories=self.categories,
            regressions=self.regressions,
        )

        return result


class BaseAiCOCOTabularInferenceModel(BaseAiCOCOInferenceModel):
    """
    Base class for defining inference model with AiCOCO format response. (tabular version)

    This class serves as a template for implementing inference functionality for various machine learning or deep learning models.
    Subclasses must override abstract methods to define model-specific behavior.

    Attributes:
        network (Callable): The inference network or model.
    """

    def __init__(self):
        super().__init__()

    def sort_tables_files(self, tables: Dict[str, Any], files: Sequence[str]):
        """
        Sort tables and files by file_names

        Args:
            tables (Dict[str, Any]): A dictionary of table information.
            files (Sequence[str]): List of input file_names.

        Returns:
            sorted_tables (Dict[str, Any]): sorted tables.
            sorted_files (Sequence[str]): sorted files.

        """

        sorted_tables = sorted(tables, key=lambda x: x["file_name"].replace("/", "_")[::-1])
        sorted_files = sorted(files, key=lambda x: x[::-1])

        for file, table in zip(sorted_files, sorted_tables):
            table_name = table["file_name"].replace("/", "_")
            if not file.endswith(table_name):
                raise ValueError(f"File names do not match table names: {file} vs {table_name}")

        return sorted_tables, sorted_files

    def check_inputs(
        self, dataframes: Sequence[pd.DataFrame], tables: Dict[str, Any], meta: Dict[str, Any]
    ):
        """
        Check dataframes and tables size and the key of meta.

        Args:
            dataframes (Sequence[DataFrame]): Sequence of dataframes correspond each tables.
            tables (Dict[str, Any]): A dictionary of table information.
            meta (Dict[str, Any]): Meta information.

        """
        assert len(dataframes) == len(tables)
        assert "window_size" in meta
        window_size = meta["window_size"]
        assert all(
            len(df) % window_size == 0 for df in dataframes
        ), f"Not all DataFrames have a length that is divisible by {window_size}"

    @abstractmethod
    def data_reader(self, files: Sequence[str], **kwargs) -> Sequence[pd.DataFrame]:
        """
        Read data for inference model.

        Args:
            files (Sequence[str]): List of input file_names.

        Returns:
            dataframes (Sequence[pd.DataFrame]): A sequence of dataframes.

        """

        raise NotImplementedError

    @abstractmethod
    def output_formatter(
        self,
        model_out: Any,
        tables: Sequence[Dict[str, Any]],
        records: Sequence[Dict[str, Any]],
        meta: Dict[str, Any],
        categories: Optional[Sequence[Dict[str, Any]]] = None,
        regressions: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """
        Abstract method to format the output of tabular inference model.
        To respond results in AiCOCO format, users should adopt output strategy specifying for various tasks.

        Args:
            model_out (Any): Inference output.
            tables (Sequence[Dict[str, Any]]): List of tables.
            records (Sequence[AiRecord]): List of inference records.
            meta (Dict[str, Any]): Additional metadata.
            categories (Optional[Sequence[Dict[str, Any]]]): List of inference categories. Default: None.
            regressions (Optional[Sequence[Dict[str, Any]]]): List of inference regressions. Default: None.

        Returns:
            Any: AiCOCO formatted output.
        """
        raise NotImplementedError

    def __call__(
        self, tables: Dict[str, Any], meta: Dict[str, Any], files: Sequence[str], **kwargs
    ) -> Any:
        """
        Run the inference model.
        """
        assert len(tables) == len(files), "`files` and `tables` should have same length."

        tables, files = self.sort_tables_files(tables, files)
        dataframes = self.data_reader(files, **kwargs)
        self.check_inputs(dataframes, tables, meta)

        out = self.preprocess(dataframes)
        out = self.inference(out)
        out = self.postprocess(out)

        result = self.output_formatter(
            out,
            tables=tables,
            meta=meta,
            dataframes=dataframes,
            categories=self.categories,
            regressions=self.regressions,
        )

        return result


class BaseAiCOCOHybridInferenceModel(BaseAiCOCOInferenceModel):
    """
    Base class for defining inference model with AiCOCO format response. (hybrid version)

    This class serves as a template for implementing inference functionality for various machine learning or deep learning models.
    Subclasses must override abstract methods to define model-specific behavior.
    """

    def __init__(self):
        super().__init__()

    def _sort_inputs(
        self,
        images: Sequence[Dict[str, Any]],
        tables: Sequence[Dict[str, Any]],
        files: Sequence[str],
    ) -> Any:
        self.images, self.tables = [], []
        image_files, table_files = [], []
        table_dict = {table["id"]: table for table in tables}
        for img in images:
            # image
            image_filename = img["file_name"].replace("/", "_")
            image_filename = next((file for file in files if file.endswith(image_filename)), None)
            if image_filename is None:
                raise ValueError(f"Image file not found: {image_filename}")
            image_files.append(image_filename)

            # table
            table_id = img["table_ids"][0]  # Assume the length is 1
            table = table_dict[table_id]
            table_filename = table["file_name"].replace("/", "_")
            table_filename = next((file for file in files if file.endswith(table_filename)), None)
            if table_filename is None:
                raise ValueError(f"Table file not found: {table_filename}")
            table_files.append(table_filename)
            self.images.append(AiImage.model_validate(img))
            self.tables.append(AiTable.model_validate(table))

        return image_files, table_files

    @abstractmethod
    def data_reader(self, image_files: Sequence[str], table_files: Sequence[str], **kwargs):
        raise NotImplementedError

    @abstractmethod
    def output_formatter(
        self,
        model_out: Any,
        images: Sequence[AiImage],
        tables: Sequence[AiTable],
        meta: Dict[str, Any],
        categories: Optional[Sequence[Dict[str, Any]]] = None,
        regressions: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """
        Abstract method to format the output of hybrid inference model.
        To respond results in AiCOCO format, users should adopt output strategy specifying for various tasks.

        Args:
            model_out (Any): Inference output.
            images (Optional[Sequence[Dict[str, Any]]]): List of images.
            tables (Sequence[Dict[str, Any]]): List of tables.
            meta (Dict[str, Any]): Additional metadata.
            categories (Optional[Sequence[Dict[str, Any]]]): List of inference categories. Default: None.
            regressions (Optional[Sequence[Dict[str, Any]]]): List of inference regressions. Default: None.

        Returns:
            Any: AiCOCO formatted output.
        """
        raise NotImplementedError

    def __call__(
        self,
        images: Sequence[Dict[str, Any]],
        tables: Sequence[Dict[str, Any]],
        meta: Dict[str, Any],
        files: Sequence[str],
        **kwargs,
    ) -> Any:
        assert len(images) == len(tables), "`tables`, and `images` should have same length."
        assert len(files) == len(images) + len(
            tables
        ), "The number of `files` should be equal to the sum of `images` and `tables`."

        image_filename, table_filename = self._sort_inputs(images, tables, files)
        x = self.data_reader(image_filename, table_filename)

        x = self.preprocess(x)
        out = self.inference(x)
        out = self.postprocess(out)

        result = self.output_formatter(
            out,
            images=self.images,
            tables=tables,
            meta=meta,
            categories=self.categories,
            regressions=self.regressions,
        )

        return result
