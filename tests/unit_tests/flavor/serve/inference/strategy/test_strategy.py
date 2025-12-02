import unittest
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from flavor.serve.inference.data_models.api import (
    AiCOCOHybridOutputDataModel,
    AiCOCOImageOutputDataModel,
    AiCOCOTabularOutputDataModel,
)
from flavor.serve.inference.data_models.functional.aicoco_data_model import (
    AiImage,
    AiMeta,
    AiRegressionItem,
    AiTable,
)
from flavor.serve.inference.strategies.aicoco_strategy import (
    AiCOCOClassificationOutputStrategy,
    AiCOCODetectionOutputStrategy,
    AiCOCOHybridClassificationOutputStrategy,
    AiCOCOHybridRegressionOutputStrategy,
    AiCOCORef,
    AiCOCORegressionOutputStrategy,
    AiCOCOSegmentationOutputStrategy,
    AiCOCOTabularClassificationOutputStrategy,
    AiCOCOTabularRegressionOutputStrategy,
    BaseAiCOCOOutputStrategy,
    BaseAiCOCOTabularOutputStrategy,
)


def get_dummy_image(
    id: str = "img1",
    file_name: str = "image.jpg",
    index: int = 0,
    category_ids: Any = None,
    regressions: Any = None,
) -> AiImage:
    """Helper to create a dummy AiImage."""
    return AiImage(
        file_name=file_name, id=id, index=index, category_ids=category_ids, regressions=regressions
    )


def get_dummy_categories() -> List[Dict[str, Any]]:
    """Helper method to create consistent test categories"""
    return [
        {"name": "Category 1", "supercategory_name": "group", "display": True},
        {"name": "Category 2", "supercategory_name": "group", "display": True},
    ]


def get_dummy_meta() -> AiMeta:
    """Helper to return a dummy AiMeta instance."""
    return AiMeta(category_ids=None, regressions=None)


class DummyImageOutputStrategy(BaseAiCOCOOutputStrategy):
    """Minimal dummy implementation for testing BaseAiCOCOOutputStrategy."""

    def __call__(self, *args, **kwargs):
        pass

    def model_to_aicoco(self, aicoco_ref: Any, model_out: Any) -> Any:
        return aicoco_ref


class TestBaseAiCOCOOutputStrategy(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = DummyImageOutputStrategy()

    def test_generate_categories(self):
        # Given - Define test categories with supercategory relationships
        categories = [
            {"name": "cat", "supercategory_name": "animal"},
            {"name": "dog", "supercategory_name": "animal"},
            {"name": "car", "supercategory_name": "vehicle"},
        ]

        # When - Generate AiCategory objects
        result = self.strategy.generate_categories(categories)

        # Then - Verify results
        # 1. Check total count (3 categories + 2 supercategories)
        self.assertEqual(len(result), 5, "Should generate 5 total categories (3 regular + 2 super)")

        # 2. Group results by type for easier verification
        regular_categories = [cat for cat in result if cat.supercategory_id is not None]
        super_categories = [cat for cat in result if cat.supercategory_id is None]

        # 3. Verify supercategories
        self.assertEqual(len(super_categories), 2, "Should create 2 supercategories")
        super_names = {cat.name for cat in super_categories}
        self.assertEqual(
            super_names, {"animal", "vehicle"}, "Should create the expected supercategories"
        )

        # 4. Verify regular categories
        self.assertEqual(len(regular_categories), 3, "Should have 3 regular categories")
        regular_names = {cat.name for cat in regular_categories}
        self.assertEqual(
            regular_names, {"cat", "dog", "car"}, "Should preserve all original category names"
        )

        # 5. Verify relationships between categories and supercategories
        super_id_by_name = {cat.name: cat.id for cat in super_categories}

        # Check each regular category points to the correct supercategory
        for cat in regular_categories:
            if cat.name in ["cat", "dog"]:
                expected_super_id = super_id_by_name["animal"]
                self.assertEqual(
                    cat.supercategory_id,
                    expected_super_id,
                    f"{cat.name} should belong to 'animal' supercategory",
                )
            elif cat.name == "car":
                expected_super_id = super_id_by_name["vehicle"]
                self.assertEqual(
                    cat.supercategory_id,
                    expected_super_id,
                    f"{cat.name} should belong to 'vehicle' supercategory",
                )

        # 6. Verify "supercategory_name" is not present in final objects
        for cat in result:
            self.assertNotIn(
                "supercategory_name",
                cat.model_dump(),
                "supercategory_name should be removed from result objects",
            )

    def test_prepare_aicoco(self):
        # Given - Set up test data
        image = get_dummy_image(id="img1", file_name="image1.jpg")
        images = [image]
        input_categories = [{"name": "cat", "supercategory_name": "animal"}]
        input_regressions = [{"name": "reg1", "superregression_name": "sup1"}]
        meta = {"category_ids": ["cat1"], "regressions": [{"regression_id": "r1", "value": 0.5}]}

        # When - Call the method under test
        aicoco_ref = self.strategy.prepare_aicoco(
            images=images,
            categories=input_categories,
            regressions=input_regressions,
            meta=meta,
        )

        # Then - Verify results
        # 1. Check return type
        self.assertIsInstance(aicoco_ref, AiCOCORef, "Should return an AiCOCORef object")

        # 2. Verify images were passed through correctly
        self.assertEqual(aicoco_ref.images, images, "Images should be preserved in the output")

        # 3. Verify meta data was properly converted
        self.assertEqual(
            aicoco_ref.meta, AiMeta(**meta), "Meta data should be converted to AiMeta object"
        )

        # 4. Verify categories processing
        # One input category results in 1 processed item + 1 extra supercategory
        self.assertEqual(
            len(aicoco_ref.categories),
            2,
            "Should generate 2 total categories (1 regular + 1 super)",
        )

        # 5. Verify regressions processing
        # One input regression results in 1 processed item + 1 extra superregression
        self.assertEqual(
            len(aicoco_ref.regressions),
            2,
            "Should generate 2 total regressions (1 regular + 1 super)",
        )

        # 6. Optional: Additional verification of category relationships
        categories = aicoco_ref.categories
        regular_categories = [cat for cat in categories if cat.supercategory_id is not None]
        super_categories = [cat for cat in categories if cat.supercategory_id is None]

        self.assertEqual(len(regular_categories), 1, "Should have 1 regular category")
        self.assertEqual(len(super_categories), 1, "Should have 1 supercategory")
        self.assertEqual(regular_categories[0].name, "cat", "Regular category should be 'cat'")
        self.assertEqual(super_categories[0].name, "animal", "Supercategory should be 'animal'")


class TestAiCOCOSegmentationOutputStrategy(unittest.TestCase):
    def setUp(self) -> None:
        # Set up common test objects
        self.strategy = AiCOCOSegmentationOutputStrategy()
        self.image = get_dummy_image(id="img1", file_name="img1.png")
        self.categories_dict = [{"name": "object", "supercategory_name": "all", "display": True}]
        self.H, self.W = 10, 10

        # Create a 4D segmentation mask with a single object (1 class, 1 slice)
        self.model_out = np.zeros((1, 1, self.H, self.W), dtype=np.int32)
        self.model_out[0, 0, 2:5, 2:5] = 1  # Create a 3x3 square object at position (2,2)

    def test_call_valid(self):
        """Test that valid segmentation output produces correct annotations and objects"""
        # When - Process a valid segmentation mask
        result = self.strategy(
            model_out=self.model_out,
            images=[self.image],
            categories=self.categories_dict,
        )

        # Then - Verify the result structure and content
        self.assertIsInstance(
            result,
            AiCOCOImageOutputDataModel,
            "Result should be an AiCOCOImageOutputDataModel instance",
        )
        # Check that all expected attributes exist
        expected_attrs = ["images", "categories", "regressions", "meta", "annotations", "objects"]
        for attr in expected_attrs:
            self.assertTrue(hasattr(result, attr), f"Result should have '{attr}' attribute")

        # Verify annotations were created for the segmentation
        self.assertGreater(len(result.annotations), 0, "Should create at least one annotation")

        # Verify exactly one object was detected from the single segment
        self.assertEqual(len(result.objects), 1, "Should create exactly one object")

    def test_validate_model_output_valid(self):
        """Test that valid model output passes validation"""
        # When/Then - Validation should not raise exceptions for valid input
        try:
            self.strategy.validate_model_output(self.model_out, self.categories_dict)
        except Exception as e:
            self.fail(f"validate_model_output() should not raise an exception for valid input: {e}")

    def test_validate_model_output_invalid_type(self):
        """Test validation rejects non-array input"""
        # When/Then - Validation should reject string input
        with self.assertRaises(TypeError, msg="Should reject non-array input"):
            self.strategy.validate_model_output("not an array", self.categories_dict)

    def test_validate_model_output_wrong_length(self):
        """Test validation rejects incorrect batch size"""
        # Given - Create input with wrong batch dimension
        invalid_out = np.zeros((2, 1, self.H, self.W), dtype=np.int32)  # Batch size of 2

        # When/Then - Validation should reject incorrect batch size
        with self.assertRaises(ValueError, msg="Should reject batch size > 1"):
            self.strategy.validate_model_output(invalid_out, self.categories_dict)

    def test_validate_model_output_wrong_ndim(self):
        """Test validation rejects incorrect dimensionality"""
        # Given - Create input with wrong number of dimensions
        invalid_out = np.zeros((self.H, self.W), dtype=np.int32)  # 2D instead of 4D

        # When/Then - Validation should reject incorrect dimensions
        with self.assertRaises(ValueError, msg="Should reject arrays with wrong ndim"):
            self.strategy.validate_model_output(invalid_out, self.categories_dict)

    def test_validate_model_output_nonint(self):
        """Test validation rejects non-integer data types"""
        # Given - Create input with float data type
        invalid_out = np.zeros((1, 1, self.H, self.W), dtype=np.float32)
        invalid_out[0, 0, 2:5, 2:5] = 1.5

        # When/Then - Validation should reject float data type
        with self.assertRaises(ValueError, msg="Should reject non-integer data types"):
            self.strategy.validate_model_output(invalid_out, self.categories_dict)

    def test_call_no_segmentation(self):
        """Test handling of empty segmentation mask (no objects detected)"""
        # Given - Create an empty segmentation mask with no objects
        model_out_empty = np.zeros((1, 1, self.H, self.W), dtype=np.int32)

        # When - Process the empty mask
        result = self.strategy(
            model_out=model_out_empty,
            images=[self.image],
            categories=self.categories_dict,
        )

        # Then - Verify no annotations or objects are created
        self.assertEqual(len(result.annotations), 0, "Should create no annotations for empty mask")
        self.assertEqual(len(result.objects), 0, "Should create no objects for empty mask")


class TestAiCOCOClassificationOutputStrategy(unittest.TestCase):
    def setUp(self) -> None:
        """Initialize the strategy instance for each test"""
        self.strategy = AiCOCOClassificationOutputStrategy()

    def test_validate_model_output_valid(self):
        """Test that valid classification output passes validation"""
        # Given - Create valid model output matching category count
        model_out = np.array([1, 0], dtype=np.int32)  # One-hot encoded class 0
        categories = get_dummy_categories()

        # When/Then - Validation should succeed without exceptions
        try:
            self.strategy.validate_model_output(model_out, categories)
        except Exception as e:
            self.fail(f"Validation should succeed for valid input but raised: {e}")

    def test_validate_model_output_invalid_type(self):
        """Test validation rejects non-array input"""
        # When/Then - Validation should reject string input
        with self.assertRaises(TypeError, msg="Should reject non-array input"):
            self.strategy.validate_model_output("not an array", get_dummy_categories())

    def test_validate_model_output_wrong_length(self):
        """Test validation rejects arrays with length not matching category count"""
        # Given - Create array shorter than category count
        model_out = np.array([1], dtype=np.int32)  # Only one value for two categories

        # When/Then - Validation should reject incorrect length
        with self.assertRaises(ValueError, msg="Should reject arrays with incorrect length"):
            self.strategy.validate_model_output(model_out, get_dummy_categories())

    def test_validate_model_output_wrong_ndim(self):
        """Test validation rejects arrays with incorrect dimensions"""
        # Given - Create 2D array instead of 1D
        model_out = np.array([[1, 0]], dtype=np.int32)  # 2D array

        # When/Then - Validation should reject incorrect dimensions
        with self.assertRaises(ValueError, msg="Should reject arrays with wrong dimensions"):
            self.strategy.validate_model_output(model_out, get_dummy_categories())

    def test_validate_model_output_nonint(self):
        """Test validation rejects non-integer data types"""
        # Given - Create array with float values
        model_out = np.array([1.5, 0.5], dtype=np.float32)

        # When/Then - Validation should reject float data type
        with self.assertRaises(ValueError, msg="Should reject non-integer data types"):
            self.strategy.validate_model_output(model_out, get_dummy_categories())

    def test_call_valid_single_image(self):
        """Test classification output handling for a single image"""
        # Given - Setup test data for single image
        model_out = np.array([1, 0], dtype=np.int32)  # Classify as class0
        image = get_dummy_image(id="img1", file_name="img1.png")
        categories = get_dummy_categories()
        meta = get_dummy_meta()

        # When - Process classification
        result = self.strategy(
            model_out=model_out, images=[image], categories=categories, meta=meta
        )

        # Then - Verify classification was applied to the image
        expected_cat_id = result.categories[0].id  # First category ID
        self.assertEqual(
            image.category_ids,
            [expected_cat_id],
            "Image should be assigned the first category (class0)",
        )
        self.assertEqual(result.meta, meta, "Metadata should be preserved")

    def test_call_valid_multiple_images(self):
        """Test classification output handling with multiple images (global classification)"""
        # Given - Setup test data with multiple images
        model_out = np.array([0, 1], dtype=np.int32)  # Classify as class1
        image1 = get_dummy_image(id="img1", file_name="img1.png")
        image2 = get_dummy_image(id="img2", file_name="img2.png")
        categories = get_dummy_categories()
        meta = get_dummy_meta()

        # When - Process classification with multiple images
        result = self.strategy(
            model_out=model_out, images=[image1, image2], categories=categories, meta=meta
        )

        # Then - Verify classification was applied to meta, not individual images
        expected_cat_id = result.categories[1].id  # Second category ID
        self.assertEqual(
            result.meta.category_ids,
            [expected_cat_id],
            "With multiple images, classification should be stored in meta",
        )
        self.assertIsNone(
            image1.category_ids,
            "Individual images should not be assigned categories with multiple images",
        )
        self.assertIsNone(
            image2.category_ids,
            "Individual images should not be assigned categories with multiple images",
        )


class TestAiCOCODetectionOutputStrategy(unittest.TestCase):
    def setUp(self) -> None:
        """Initialize common test fixtures for each test method"""
        # Create strategy instance
        self.strategy = AiCOCODetectionOutputStrategy()

        # Setup test image
        self.image = {
            "file_name": "test.jpg",
            "id": "img_1",
            "index": 0,
            "category_ids": None,
            "regressions": None,
        }

        # Setup test categories (cat and dog)
        self.categories = [
            {"name": "cat", "supercategory_name": "animal", "display": True},
            {"name": "dog", "supercategory_name": "animal", "display": True},
        ]

        # Empty regressions list
        self.regressions = []

        # Default model output with one detection (dog class)
        self.model_out = {
            "bbox_pred": [[10, 20, 30, 40]],  # [x, y, width, height]
            "cls_pred": [[1, 0]],  # One-hot encoding: [dog, cat]
        }

    def test_call_valid(self):
        """Test that valid detection output produces correct annotations and objects"""
        # When - Process a valid detection
        result = self.strategy(
            model_out=self.model_out,
            images=[self.image],
            categories=self.categories,
            regressions=self.regressions,
        )

        # Then - Verify the result structure and content
        # Check that all expected attributes exist
        expected_attrs = ["images", "categories", "regressions", "meta", "annotations", "objects"]
        for attr in expected_attrs:
            self.assertTrue(hasattr(result, attr), f"Result should have '{attr}' attribute")

        # Verify exactly one annotation was created
        self.assertEqual(len(result.annotations), 1, "Should create exactly one annotation")

        # Verify the annotation has correct bounding box and image ID
        annot = result.annotations[0]
        self.assertEqual(
            annot.bbox, [[10, 20, 30, 40]], "Annotation should have correct bbox coordinates"
        )
        self.assertEqual(
            annot.image_id, self.image["id"], "Annotation should reference correct image ID"
        )

        # Verify exactly one object was created
        self.assertEqual(len(result.objects), 1, "Should create exactly one object")

        # Verify the object has category information
        obj = result.objects[0]
        self.assertTrue(obj.category_ids, "Object should have category IDs assigned")

    def test_validate_model_output_missing_bbox_pred(self):
        """Test validation rejects model output missing bbox_pred"""
        # Given - Create model output without bbox_pred
        bad_model_out = {"cls_pred": [[1, 0]]}

        # When/Then - Validation should reject missing bbox_pred
        with self.assertRaises(KeyError, msg="Should reject model output missing bbox_pred"):
            self.strategy.validate_model_output(bad_model_out, self.categories)

    def test_validate_model_output_missing_cls_pred(self):
        """Test validation rejects model output missing cls_pred"""
        # Given - Create model output without cls_pred
        bad_model_out = {"bbox_pred": [[10, 20, 30, 40]]}

        # When/Then - Validation should reject missing cls_pred
        with self.assertRaises(KeyError, msg="Should reject model output missing cls_pred"):
            self.strategy.validate_model_output(bad_model_out, self.categories)

    def test_validate_model_output_length_mismatch(self):
        """Test validation rejects model output with length mismatch between bbox_pred and cls_pred"""
        # Given - Create model output with more bboxes than class predictions
        bad_model_out = {
            "bbox_pred": [[10, 20, 30, 40], [50, 60, 70, 80]],  # Two bboxes
            "cls_pred": [[1, 0]],  # One class prediction
        }

        # When/Then - Validation should reject length mismatch
        with self.assertRaises(
            ValueError, msg="Should reject length mismatch between bbox_pred and cls_pred"
        ):
            self.strategy.validate_model_output(bad_model_out, self.categories)

    def test_validate_model_output_cls_pred_inner_length(self):
        """Test validation rejects class predictions with wrong number of classes"""
        # Given - Create model output with fewer classes than expected
        bad_model_out = {
            "bbox_pred": [[10, 20, 30, 40]],
            "cls_pred": [[1]],  # Only one class value instead of two
        }

        # When/Then - Validation should reject incorrect class vector length
        with self.assertRaises(
            ValueError, msg="Should reject cls_pred with wrong number of classes"
        ):
            self.strategy.validate_model_output(bad_model_out, self.categories)

    def test_model_to_aicoco_with_confidence_and_regression(self):
        """Test detection with confidence scores and regression values"""
        # Given - Create model output with confidence and regression
        model_out = {
            "bbox_pred": [[15, 25, 35, 45]],
            "cls_pred": [[1, 0]],  # Dog class
            "confidence_score": [0.9],  # 90% confidence
            "regression_value": [[0.5, 0.7]],  # Regression values
        }

        # Add a regression type
        regressions = [{"name": "size", "superregression_name": None}]

        # When - Process detection with confidence and regression
        result = self.strategy(
            model_out=model_out,
            images=[self.image],
            categories=self.categories,
            regressions=regressions,
        )

        # Then - Verify confidence and regression data
        self.assertEqual(len(result.annotations), 1, "Should create one annotation")
        self.assertEqual(len(result.objects), 1, "Should create one object")

        # Verify confidence score
        obj = result.objects[0]
        self.assertTrue(hasattr(obj, "confidence"), "Object should have confidence attribute")
        self.assertEqual(obj.confidence, 0.9, "Object should have correct confidence value")

        # Verify regression data
        self.assertIsNotNone(obj.regressions, "Object should have regression data")

    def test_generate_annotations_objects_no_detection(self):
        """Test handling when no objects are detected (all class scores zero)"""
        # Given - Create model output with no detections (all zeros in cls_pred)
        model_out = {"bbox_pred": [[10, 20, 30, 40]], "cls_pred": [[0, 0]]}  # No class detected

        # When - Process detection with no objects
        result = self.strategy(
            model_out=model_out,
            images=[self.image],
            categories=self.categories,
            regressions=self.regressions,
        )

        # Then - Verify no annotations or objects are created
        self.assertEqual(
            len(result.annotations), 0, "Should create no annotations when no objects detected"
        )
        self.assertEqual(
            len(result.objects), 0, "Should create no objects when no objects detected"
        )


class TestAiCOCORegressionOutputStrategy(unittest.TestCase):
    def setUp(self) -> None:
        """Initialize common test fixtures for each test method"""
        # Create strategy instance
        self.strategy = AiCOCORegressionOutputStrategy()

        # Setup test image
        self.image = get_dummy_image(id="img_1", file_name="test.jpg")

        # Setup single regression definition
        self.regressions_dict = [{"name": "size", "superregression_name": None}]

    def test_validate_model_output_valid_1d(self):
        """Test validation accepts valid 1D numpy array"""
        # Given - Create a valid 1D regression value
        model_out = np.array([0.5])

        # When/Then - Validation should succeed without exceptions
        try:
            self.strategy.validate_model_output(model_out)
        except Exception as e:
            self.fail(f"Validation should succeed for valid 1D array but raised: {e}")

    def test_validate_model_output_valid_4d(self):
        """Test validation accepts valid 4D numpy array (e.g., from CNN)"""
        # Given - Create a valid 4D regression value (common from convolutional models)
        model_out = np.zeros((1, 1, 1, 1))

        # When/Then - Validation should succeed without exceptions
        try:
            self.strategy.validate_model_output(model_out)
        except Exception as e:
            self.fail(f"Validation should succeed for valid 4D array but raised: {e}")

    def test_validate_model_output_wrong_type(self):
        """Test validation rejects non-numpy array input"""
        # Given - Create input with wrong type (Python list instead of numpy array)
        model_out = [0.5]

        # When/Then - Validation should reject incorrect type
        with self.assertRaises(TypeError, msg="Should reject non-numpy array input"):
            self.strategy.validate_model_output(model_out)

    def test_validate_model_output_wrong_ndim(self):
        """Test validation rejects numpy arrays with unsupported dimensions"""
        # Given - Create 2D array which is not in the supported formats
        model_out = np.zeros((2, 2))

        # When/Then - Validation should reject unsupported dimensions
        with self.assertRaises(ValueError, msg="Should reject arrays with unsupported dimensions"):
            self.strategy.validate_model_output(model_out)

    def test_call_single_image(self):
        """Test regression output handling for a single image"""
        # Given - Setup regression value for a single image
        model_out = np.array([0.75])

        # When - Process regression for single image
        result = self.strategy(
            model_out=model_out,
            images=[self.image],
            regressions=self.regressions_dict,
        )

        # Then - Verify regression was attached to the image
        self.assertIsInstance(
            result,
            AiCOCOImageOutputDataModel,
            "Result should be an AiCOCOImageOutputDataModel instance",
        )
        updated_image = result.images[0]
        self.assertIsNotNone(updated_image.regressions, "Image should have regression values")
        self.assertEqual(
            len(updated_image.regressions), 1, "Should have exactly one regression value"
        )

        # Check regression value
        reg_item = updated_image.regressions[0]
        self.assertEqual(reg_item.value, 0.75, "Regression value should match input")

        # Verify meta has no regressions (since they're on the image)
        meta_regressions = (
            result.meta.get("regressions")
            if isinstance(result.meta, dict)
            else result.meta.regressions
        )
        self.assertTrue(
            meta_regressions in (None, []),
            "Meta should not have regressions when they're on the image",
        )

    def test_call_multiple_images(self):
        """Test regression output handling with multiple images (global regression)"""
        # Given - Setup regression value and multiple images
        model_out = np.array([0.85])
        image1 = get_dummy_image(id="img1", file_name="img1.jpg")
        image2 = get_dummy_image(id="img2", file_name="img2.jpg")

        # When - Process regression with multiple images
        result = self.strategy(
            model_out=model_out,
            images=[image1, image2],
            regressions=self.regressions_dict,
        )

        # Then - Verify regression was attached to the meta, not images
        meta = result.meta
        reg_list = meta.get("regressions") if isinstance(meta, dict) else meta.regressions
        self.assertIsNotNone(reg_list, "Meta should have regression values with multiple images")
        self.assertEqual(len(reg_list), 1, "Should have exactly one regression value")

        # Check regression value
        reg_item = reg_list[0]
        self.assertEqual(reg_item.value, 0.85, "Regression value should match input")

        # Verify individual images don't have regressions
        for img in result.images:
            self.assertIsNone(
                img.regressions,
                "Individual images should not have regressions with multiple images",
            )


class DummyTabularOutputStrategy(BaseAiCOCOTabularOutputStrategy):
    def __call__(self, *args, **kwargs):
        pass

    def model_to_aicoco(self, aicoco_ref: Any, model_out: Any, **kwargs) -> Any:
        return aicoco_ref


class TestBaseAiCOCOTabularOutputStrategy(unittest.TestCase):
    def setUp(self) -> None:
        """Initialize the strategy instance for each test"""
        self.strategy = DummyTabularOutputStrategy()

    def test_generate_records(self):
        """Test generation of records from dataframes"""
        # Given - Create test dataframe and metadata
        df = pd.DataFrame({"a": range(5)})
        meta = {"window_size": 2}
        table = AiTable(id="table1", file_name="dummy.csv")

        # When - Generate records
        records = self.strategy.generate_records([df], [table], meta)

        # Then - Verify records are correctly generated
        self.assertEqual(
            len(records),
            2,
            "Should generate 2 records for a dataframe with 5 rows and window size 2",
        )
        expected_indexes = [[0, 1], [2, 3]]
        for rec, exp in zip(records, expected_indexes):
            self.assertEqual(rec.table_id, "table1", "Record should reference the correct table")
            self.assertEqual(rec.row_indexes, exp, "Record should contain the expected row indexes")

    def test_prepare_aicoco(self):
        """Test preparation of AiCOCO tabular format from input data"""
        # Given - Create test input data
        tables_input = [{"id": "table1", "file_name": "dummy.csv"}]
        meta = {"window_size": 2}
        df = pd.DataFrame({"a": range(4)})
        dataframes = [df]
        categories_input = [{"name": "Category1"}]
        regressions_input = [{"name": "Regression1"}]

        # When - Prepare AiCOCO format
        aicoco_output = self.strategy.prepare_aicoco(
            tables=tables_input,
            meta=meta,
            dataframes=dataframes,
            categories=categories_input,
            regressions=regressions_input,
        )

        # Then - Verify AiCOCO output structure
        self.assertIsInstance(
            aicoco_output,
            AiCOCOTabularOutputDataModel,
            "Output should be an AiCOCOTabularOutputDataModel instance",
        )

        # Verify tables
        self.assertEqual(len(aicoco_output.tables), 1, "Should have 1 table")
        self.assertEqual(aicoco_output.tables[0].id, "table1", "Table ID should match input")

        # Verify records
        self.assertEqual(
            len(aicoco_output.records), 2, "Should generate 2 records with window size 2"
        )
        expected_indexes = [[0, 1], [2, 3]]
        for rec, exp in zip(aicoco_output.records, expected_indexes):
            self.assertEqual(rec.table_id, "table1", "Record should reference correct table")
            self.assertEqual(rec.row_indexes, exp, "Record should contain expected row indexes")

        # Verify categories
        self.assertEqual(len(aicoco_output.categories), 1, "Should have 1 category")
        self.assertEqual(
            aicoco_output.categories[0].name, "Category1", "Category name should match input"
        )

        # Verify regressions
        self.assertEqual(len(aicoco_output.regressions), 1, "Should have 1 regression")
        self.assertEqual(
            aicoco_output.regressions[0].name, "Regression1", "Regression name should match input"
        )

        # Verify metadata
        self.assertEqual(aicoco_output.meta.window_size, 2, "Window size should match input")


class TestAiCOCOTabularClassificationOutputStrategy(unittest.TestCase):
    def setUp(self) -> None:
        """Initialize the strategy instance for each test"""
        self.strategy = AiCOCOTabularClassificationOutputStrategy()

    def test_validate_model_output_valid(self):
        """Test that valid classification output passes validation"""
        # Given - Create valid model output matching category count
        model_out = np.array([[1, 0], [0, 1]], dtype=np.int32)
        categories = get_dummy_categories()

        # When/Then - Validation should succeed without exceptions
        try:
            self.strategy.validate_model_output(model_out, categories)
        except Exception as e:
            self.fail(f"Validation should succeed for valid input but raised: {e}")

    def test_validate_model_output_invalid_type(self):
        """Test validation rejects non-array input"""
        # Given - Create non-array input
        model_out = [[1, 0], [0, 1]]

        # When/Then - Validation should reject non-array input
        with self.assertRaises(TypeError, msg="Should reject non-array input"):
            self.strategy.validate_model_output(model_out, get_dummy_categories())

    def test_validate_model_output_invalid_dimension(self):
        """Test validation rejects arrays with incorrect dimensions"""
        # Given - Create array with wrong dimensions
        model_out = np.array([[[1]]], dtype=np.int32)

        # When/Then - Validation should reject incorrect dimensions
        with self.assertRaises(ValueError, msg="Should reject arrays with wrong dimensions"):
            self.strategy.validate_model_output(model_out, get_dummy_categories())

    def test_validate_model_output_mismatched_last_dim(self):
        """Test validation rejects arrays with incorrect class count"""
        # Given - Create array with more classes than categories
        model_out = np.array([[1, 0, 1]], dtype=np.int32)

        # When/Then - Validation should reject mismatched class count
        with self.assertRaises(ValueError, msg="Should reject arrays with incorrect class count"):
            self.strategy.validate_model_output(model_out, get_dummy_categories())

    def test_validate_model_output_non_binary(self):
        """Test validation rejects non-binary values"""
        # Given - Create array with non-binary values
        model_out = np.array([[2, 0]], dtype=np.int32)

        # When/Then - Validation should reject non-binary values
        with self.assertRaises(ValueError, msg="Should reject arrays with non-binary values"):
            self.strategy.validate_model_output(model_out, get_dummy_categories())

    def test_model_to_aicoco(self):
        """Test conversion from model output to AiCOCO format"""
        # Given - Setup test data
        tables = [{"id": "table1", "file_name": "dummy.csv"}]
        dataframes = [pd.DataFrame({"a": [1]})]
        meta = {"window_size": 1}
        categories = get_dummy_categories()
        aicoco_ref = self.strategy.prepare_aicoco(tables, meta, dataframes, categories)
        model_out = np.array([[1, 0]], dtype=np.int32)

        # When - Convert model output to AiCOCO
        out_ref = self.strategy.model_to_aicoco(aicoco_ref, model_out)

        # Then - Verify conversion results
        self.assertIsInstance(
            out_ref,
            AiCOCOTabularOutputDataModel,
            "Result should be an AiCOCOTabularOutputDataModel instance",
        )
        self.assertEqual(
            out_ref.records[0].category_ids[0],
            out_ref.categories[0].id,
            "Record should be assigned the first category",
        )

    def test_call_success(self):
        """Test successful classification of multiple tables"""
        # Given - Setup test data for multiple tables
        dataframes = [pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]})]
        tables = [
            {"id": "table1", "file_name": "table1.csv"},
            {"id": "table2", "file_name": "table2.csv"},
        ]
        meta = {"window_size": 1}
        categories = get_dummy_categories()
        model_out = np.array([[1, 0], [0, 1]], dtype=np.int32)

        # When - Process classification
        aicoco_out = self.strategy(
            model_out=model_out,
            tables=tables,
            dataframes=dataframes,
            categories=categories,
            meta=meta,
        )

        # Then - Verify classification results
        records = aicoco_out.records
        output_categories = aicoco_out.categories

        self.assertEqual(len(records), 2, "Should create records for both tables")
        self.assertEqual(
            output_categories[0].name,
            categories[0]["name"],
            "First category name should match input",
        )
        self.assertEqual(
            output_categories[1].name,
            categories[1]["name"],
            "Second category name should match input",
        )
        self.assertEqual(
            records[0].category_ids[0],
            output_categories[0].id,
            "First record should be assigned the first category",
        )
        self.assertEqual(
            records[1].category_ids[0],
            output_categories[1].id,
            "Second record should be assigned the second category",
        )

    def test_call_mismatched_records(self):
        """Test error handling when model output doesn't match record count"""
        # Given - Setup test data with mismatched record count
        dataframes = [pd.DataFrame({"a": [1]})]
        tables = [{"id": "table1", "file_name": "table1.csv"}]
        meta = {"window_size": 2}  # Will generate 0 records with this small dataframe
        categories = get_dummy_categories()
        model_out = np.array([[1, 0], [0, 1]], dtype=np.int32)  # 2 classifications

        # When/Then - Strategy should reject mismatched record count
        with self.assertRaises(ValueError, msg="Should reject mismatched record count"):
            _ = self.strategy(
                model_out=model_out,
                tables=tables,
                dataframes=dataframes,
                categories=categories,
                meta=meta,
            )


class TestAiCOCOTabularRegressionOutputStrategy(unittest.TestCase):
    def setUp(self) -> None:
        """Initialize the strategy instance and test data for each test"""
        self.strategy = AiCOCOTabularRegressionOutputStrategy()
        self.tables = [
            {"id": "table1", "file_name": "table1.csv"},
            {"id": "table2", "file_name": "table2.csv"},
        ]
        self.dataframes = [pd.DataFrame({"col": [1]}), pd.DataFrame({"col": [2]})]
        self.meta = {"window_size": 1}
        self.regressions = [
            {"name": "reg1", "display": True},
            {"name": "reg2", "display": True},
        ]

    def test_validate_model_output_valid(self):
        """Test that valid regression output passes validation"""
        # Given - Create valid model output
        model_out = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        # When/Then - Validation should succeed without exceptions
        try:
            self.strategy.validate_model_output(model_out)
        except Exception as e:
            self.fail(f"Validation should succeed for valid input but raised: {e}")

    def test_validate_model_output_wrong_type(self):
        """Test validation rejects non-array input"""
        # Given - Create non-array input
        model_out = [[1.0, 2.0]]

        # When/Then - Validation should reject non-array input
        with self.assertRaises(TypeError, msg="Should reject non-array input"):
            self.strategy.validate_model_output(model_out)

    def test_validate_model_output_too_high_dim(self):
        """Test validation rejects arrays with too many dimensions"""
        # Given - Create array with too many dimensions
        model_out = np.zeros((1, 1, 1), dtype=np.float32)

        # When/Then - Validation should reject arrays with too many dimensions
        with self.assertRaises(ValueError, msg="Should reject arrays with too many dimensions"):
            self.strategy.validate_model_output(model_out)

    def test_validate_model_output_inf_values(self):
        """Test validation rejects arrays with infinite values"""
        # Given - Create array with infinite values
        model_out = np.array([[np.inf]], dtype=np.float32)

        # When/Then - Validation should reject arrays with infinite values
        with self.assertRaises(ValueError, msg="Should reject arrays with infinite values"):
            self.strategy.validate_model_output(model_out)

    def test_call_single_regression(self):
        """Test regression output handling with a single regression target"""
        # Given - Setup test data for single regression
        model_out = np.array([[0.5], [0.7]], dtype=np.float32)  # 2 records × 1 regression

        # When - Process regression with single target
        result = self.strategy(
            model_out=model_out,
            tables=self.tables,
            dataframes=self.dataframes,
            regressions=[{"name": "reg1", "display": True}],
            meta=self.meta,
        )

        # Then - Verify regression was applied correctly
        self.assertIsInstance(
            result,
            AiCOCOTabularOutputDataModel,
            "Result should be an AiCOCOTabularOutputDataModel instance",
        )
        self.assertEqual(len(result.records), 2, "Should have created 2 records")
        for i, record in enumerate(result.records):
            self.assertEqual(
                len(record.regressions), 1, f"Record {i} should have 1 regression value"
            )
            self.assertIsInstance(
                record.regressions[0],
                AiRegressionItem,
                f"Record {i} regression should be an AiRegressionItem",
            )

    def test_call_multiple_regression(self):
        """Test regression output handling with multiple regression targets"""
        # Given - Setup test data for multiple regressions
        model_out = np.array(
            [[1.1, 2.2], [3.3, 4.4]], dtype=np.float32
        )  # 2 records × 2 regressions

        # When - Process regression with multiple targets
        result = self.strategy(
            model_out=model_out,
            tables=self.tables,
            dataframes=self.dataframes,
            regressions=self.regressions,
            meta=self.meta,
        )

        # Then - Verify regression values were applied correctly
        self.assertEqual(len(result.records), 2, "Should have created 2 records")
        for i, record in enumerate(result.records):
            self.assertEqual(
                len(record.regressions), 2, f"Record {i} should have 2 regression values"
            )
            for j, reg in enumerate(record.regressions):
                self.assertIsInstance(
                    reg,
                    AiRegressionItem,
                    f"Record {i} regression {j} should be an AiRegressionItem",
                )
                # Could add assertions to verify values match model_out if needed

    def test_model_output_length_mismatch(self):
        """Test error handling when model output doesn't match record count"""
        # Given - Setup test data with mismatched record count
        model_out = np.array(
            [[1.0], [2.0], [3.0]], dtype=np.float32
        )  # 3 outputs but only 2 records

        # When/Then - Strategy should reject mismatched record count
        with self.assertRaises(ValueError, msg="Should reject mismatched record count") as ctx:
            self.strategy(
                model_out=model_out,
                tables=self.tables,
                dataframes=self.dataframes,
                regressions=[{"name": "reg1", "display": True}],
                meta=self.meta,
            )
        self.assertIn(
            "number of records is not matched",
            str(ctx.exception),
            "Error message should mention record count mismatch",
        )


class TestAiCOCOHybridClassificationOutputStrategy(unittest.TestCase):
    def setUp(self) -> None:
        """Initialize the strategy instance and test data for each test"""
        self.strategy = AiCOCOHybridClassificationOutputStrategy()
        self.images = [
            get_dummy_image(id="img1", file_name="img1.jpg"),
            get_dummy_image(id="img2", file_name="img2.jpg"),
        ]
        self.model_out = np.array([1, 0], dtype=np.int32)
        self.categories = [
            {"name": "cat", "supercategory_name": "animal", "display": True},
            {"name": "dog", "supercategory_name": "animal", "display": True},
        ]
        self.tables = [
            {"id": "table1", "file_name": "table1.csv"},
            {"id": "table2", "file_name": "table2.csv"},
        ]

    def test_call_valid(self):
        """Test classification output handling with hybrid strategy"""
        # Given - Setup is done in setUp method

        # When - Process classification with hybrid strategy
        result = self.strategy(
            model_out=self.model_out,
            images=self.images,
            tables=self.tables,
            categories=self.categories,
        )

        # Then - Verify hybrid output structure contains expected components
        self.assertIsInstance(
            result,
            AiCOCOHybridOutputDataModel,
            "Result should be an AiCOCOHybridOutputDataModel instance",
        )
        self.assertTrue(hasattr(result, "images"), "Result should contain images")
        self.assertTrue(hasattr(result, "categories"), "Result should contain categories")
        self.assertTrue(hasattr(result, "tables"), "Result should contain tables")
        self.assertEqual(len(result.tables), 2, "Should include all provided tables")
        self.assertIsInstance(
            result.tables[0], AiTable, "Tables should be converted to AiTable objects"
        )
        self.assertEqual(
            result.tables[0].file_name, "table1.csv", "Table file name should be preserved"
        )

    def test_model_to_aicoco_adds_tables(self):
        """Test that model_to_aicoco correctly adds tables to AiCOCO output"""
        # Given - Prepare base AiCOCO reference without tables
        aicoco_ref = self.strategy.prepare_aicoco(images=self.images, categories=self.categories)

        # When - Convert model output to AiCOCO and add tables
        result = self.strategy.model_to_aicoco(
            aicoco_ref=aicoco_ref,
            model_out=self.model_out,
            tables=self.tables,
        )

        # Then - Verify tables were added correctly
        self.assertTrue(hasattr(result, "tables"), "Result should contain tables attribute")
        self.assertEqual(len(result.tables), 2, "Should include all provided tables")
        for i, table in enumerate(result.tables):
            self.assertIsInstance(table, AiTable, f"Table {i} should be an AiTable instance")
            self.assertEqual(
                table.file_name, self.tables[i]["file_name"], f"Table {i} should preserve file name"
            )

    def test_validate_model_output_invalid_type(self):
        """Test validation rejects non-array input"""
        # Given - Create non-array input
        bad_output = "not an array"

        # When/Then - Validation should reject non-array input
        with self.assertRaises(TypeError, msg="Should reject non-array input"):
            self.strategy.validate_model_output(bad_output, self.categories)

    def test_validate_model_output_wrong_length(self):
        """Test validation rejects arrays with length not matching category count"""
        # Given - Create array with incorrect length
        wrong_length_output = np.array([1], dtype=np.int32)

        # When/Then - Validation should reject incorrect length
        with self.assertRaises(ValueError, msg="Should reject arrays with incorrect length"):
            self.strategy.validate_model_output(wrong_length_output, self.categories)

    def test_validate_model_output_wrong_dtype(self):
        """Test validation rejects arrays with non-integer data types"""
        # Given - Create array with non-integer data type
        bad_output = np.array([0.5, 1.5], dtype=np.float32)

        # When/Then - Validation should reject non-integer data types
        with self.assertRaises(ValueError, msg="Should reject arrays with non-integer data types"):
            self.strategy.validate_model_output(bad_output, self.categories)


class TestAiCOCOHybridRegressionOutputStrategy(unittest.TestCase):
    def setUp(self) -> None:
        """Initialize the strategy instance and test data for each test"""
        self.strategy = AiCOCOHybridRegressionOutputStrategy()
        self.images = [
            get_dummy_image(id="img1", file_name="img1.jpg"),
        ]
        self.model_out = np.array([1.0], dtype=np.float32)
        self.regressions = [
            {"name": "reg1", "display": True},
        ]
        self.tables = [
            {"id": "table1", "file_name": "table1.csv"},
        ]

    def test_call_valid(self):
        """Test regression output handling with hybrid strategy"""
        # Given - Setup is done in setUp method

        # When - Process regression with hybrid strategy
        result = self.strategy(
            model_out=self.model_out,
            images=self.images,
            tables=self.tables,
            regressions=self.regressions,
        )

        # Then - Verify hybrid output structure contains expected components
        self.assertIsInstance(
            result,
            AiCOCOHybridOutputDataModel,
            "Result should be an AiCOCOHybridOutputDataModel instance",
        )
        self.assertTrue(hasattr(result, "images"), "Result should contain images")
        self.assertTrue(hasattr(result, "regressions"), "Result should contain regressions")
        self.assertTrue(hasattr(result, "tables"), "Result should contain tables")
        self.assertEqual(len(result.tables), 1, "Should include all provided tables")
        self.assertIsInstance(
            result.tables[0], AiTable, "Tables should be converted to AiTable objects"
        )
        self.assertEqual(
            result.tables[0].file_name, "table1.csv", "Table file name should be preserved"
        )

    def test_model_to_aicoco_adds_tables(self):
        """Test that model_to_aicoco correctly adds tables to AiCOCO output"""
        # Given - Prepare base AiCOCO reference without tables
        aicoco_ref = self.strategy.prepare_aicoco(images=self.images, regressions=self.regressions)

        # When - Convert model output to AiCOCO and add tables
        result = self.strategy.model_to_aicoco(
            aicoco_ref=aicoco_ref,
            model_out=self.model_out,
            tables=self.tables,
        )

        # Then - Verify tables were added correctly
        self.assertTrue(hasattr(result, "tables"), "Result should contain tables attribute")
        self.assertEqual(len(result.tables), 1, "Should include all provided tables")
        for i, table in enumerate(result.tables):
            self.assertIsInstance(table, AiTable, f"Table {i} should be an AiTable instance")
            self.assertEqual(
                table.file_name, self.tables[i]["file_name"], f"Table {i} should preserve file name"
            )

    def test_validate_model_output_invalid_type(self):
        """Test validation rejects non-array input"""
        # Given - Create non-array input
        bad_output = "not an array"

        # When/Then - Validation should reject non-array input
        with self.assertRaises(TypeError, msg="Should reject non-array input"):
            self.strategy.validate_model_output(bad_output)

    def test_validate_model_output_wrong_length(self):
        """Test validation rejects arrays with incorrect dimensions"""
        # Given - Create array with incorrect dimensions
        wrong_length_output = np.array([[1.0], [2.0]], dtype=np.float32)

        # When/Then - Validation should reject incorrect dimensions
        with self.assertRaises(ValueError, msg="Should reject arrays with incorrect dimensions"):
            self.strategy.validate_model_output(wrong_length_output)


if __name__ == "__main__":
    unittest.main()
