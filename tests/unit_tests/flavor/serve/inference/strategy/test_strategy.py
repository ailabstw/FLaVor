import unittest
from typing import Any, Dict, List

import numpy as np

from flavor.serve.inference.data_models.functional.aicoco_data_model import (
    AiImage,
    AiMeta,
)

# Import the necessary classes from your modules
from flavor.serve.inference.strategies.aicoco_strategy import (
    AiCOCOClassificationOutputStrategy,
    AiCOCODetectionOutputStrategy,
    AiCOCOSegmentationOutputStrategy,
    BaseAiCOCOOutputStrategy,
    generate,
    set_global_seed,
)


# Create a dummy subclass of BaseAiCOCOOutputStrategy to implement abstract methods.
class DummyStrategy(BaseAiCOCOOutputStrategy):
    def __call__(self, *args, **kwargs):
        # Minimal implementation for testing.
        pass

    def model_to_aicoco(self, aicoco_ref, model_out):
        # Minimal implementation: simply return the reference.
        return aicoco_ref


@unittest.skip
class TestBaseAiCOCOOutputStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = DummyStrategy()

    def test_generate_categories(self):
        # Reset the global seed to get deterministic IDs.
        set_global_seed(100)
        categories_input = [
            {"name": "cat1", "supercategory_name": "super1"},
            {"name": "cat2", "supercategory_name": "super1"},
            {"name": "cat3", "supercategory_name": "super2"},
        ]
        result = self.strategy.generate_categories(categories_input)
        # We expect the original three categories plus two additional supercategory entries.
        self.assertEqual(len(result), 5)

        # The first three entries are the processed input categories.
        cat1, cat2, cat3 = result[0:3]

        # Convert models to dicts for easier inspection.
        for cat in [cat1, cat2, cat3]:
            cat_dict = cat.model_dump()
            self.assertNotIn("supercategory_name", cat_dict)
            self.assertIn("id", cat_dict)
            self.assertIn("name", cat_dict)
            self.assertIn("supercategory_id", cat_dict)

        # Check that categories with the same supercategory share the same supercategory_id.
        self.assertEqual(
            cat1.model_dump()["supercategory_id"], cat2.model_dump()["supercategory_id"]
        )
        self.assertNotEqual(
            cat1.model_dump()["supercategory_id"], cat3.model_dump()["supercategory_id"]
        )

        # The last two entries are the supercategory definitions.
        supercat1, supercat2 = result[3], result[4]
        for sup in [supercat1, supercat2]:
            sup_dict = sup.model_dump()
            self.assertIsNone(sup_dict.get("supercategory_id"))
            self.assertIn("id", sup_dict)
            self.assertIn("name", sup_dict)

        super_ids = {supercat1.model_dump()["id"], supercat2.model_dump()["id"]}
        self.assertIn(cat1.model_dump()["supercategory_id"], super_ids)
        self.assertIn(cat3.model_dump()["supercategory_id"], super_ids)

    def test_generate_regressions(self):
        set_global_seed(200)
        regressions_input = [
            {"name": "reg1", "superregression_name": "sreg1"},
            {"name": "reg2", "superregression_name": "sreg1"},
            {"name": "reg3", "superregression_name": "sreg2"},
        ]
        result = self.strategy.generate_regressions(regressions_input)
        # Expecting three processed regressions plus two extra entries for the superregressions.
        self.assertEqual(len(result), 5)

        reg1, reg2, reg3 = result[0:3]
        for reg in [reg1, reg2, reg3]:
            reg_dict = reg.model_dump()
            self.assertNotIn("superregression_name", reg_dict)
            self.assertIn("id", reg_dict)
            self.assertIn("name", reg_dict)
            self.assertIn("superregression_id", reg_dict)

        # The first two regressions should share the same superregression_id.
        self.assertEqual(
            reg1.model_dump()["superregression_id"], reg2.model_dump()["superregression_id"]
        )
        self.assertNotEqual(
            reg1.model_dump()["superregression_id"], reg3.model_dump()["superregression_id"]
        )

        # The last two entries are the superregression definitions.
        sreg1, sreg2 = result[3], result[4]
        for sreg in [sreg1, sreg2]:
            sreg_dict = sreg.model_dump()
            self.assertIsNone(sreg_dict.get("superregression_id"))
            self.assertIn("id", sreg_dict)
            self.assertIn("name", sreg_dict)

        super_ids = {sreg1.model_dump()["id"], sreg2.model_dump()["id"]}
        self.assertIn(reg1.model_dump()["superregression_id"], super_ids)
        self.assertIn(reg3.model_dump()["superregression_id"], super_ids)

    def test_prepare_aicoco(self):
        # Prepare dummy inputs for images, categories, regressions, and meta.
        images = [{"id": "img1", "file_name": "image1.jpg"}]
        categories_input = [{"name": "cat1", "supercategory_name": "super1"}]
        regressions_input = [{"name": "reg1", "superregression_name": "sreg1"}]
        meta = {"category_ids": [1, 2, 3], "regressions": ["r1", "r2"]}

        aicoco_ref = self.strategy.prepare_aicoco(images, categories_input, regressions_input, meta)

        # Check that the returned dictionary has the expected keys.
        self.assertIsInstance(aicoco_ref, dict)
        self.assertIn("images", aicoco_ref)
        self.assertIn("categories", aicoco_ref)
        self.assertIn("regressions", aicoco_ref)
        self.assertIn("meta", aicoco_ref)

        self.assertEqual(aicoco_ref["images"], images)
        self.assertEqual(aicoco_ref["meta"], meta)

        # Check that the categories and regressions were processed.
        # For one input category, we expect 2 entries (the category and its supercategory).
        self.assertTrue(isinstance(aicoco_ref["categories"], list))
        self.assertEqual(len(aicoco_ref["categories"]), 2)
        # Similarly for regressions.
        self.assertTrue(isinstance(aicoco_ref["regressions"], list))
        self.assertEqual(len(aicoco_ref["regressions"]), 2)

    def test_generate_function_with_global_seed(self):
        # Test that when GLOBAL_SEED is set, the generate() function produces IDs of the correct length.
        set_global_seed(500)
        id1 = generate()
        id2 = generate()
        self.assertNotEqual(id1, id2)
        self.assertEqual(len(id1), 21)
        self.assertEqual(len(id2), 21)


@unittest.skip
class TestAiCOCOSegmentationOutputStrategy(unittest.TestCase):
    def setUp(self):
        # Reset our global counter before each test for predictability.
        global global_counter
        global_counter = 0

        # Create an instance of the strategy.
        self.strategy = AiCOCOSegmentationOutputStrategy()

        # Create a dummy image (with an "id" attribute).
        self.image = AiImage(
            file_name="img1.png", id="img1", index=0, category_ids=None, regressions=None
        )

        # Note: The strategy's generate_categories() method will use these dictionaries
        # and then call AiCategory.model_validate() to create dummy AiCategory objects.
        self.categories_dict = [{"name": "object", "supercategory_name": "all", "display": True}]

        # Define a valid segmentation mask.
        # Expected shape: (num_classes, slices, H, W).
        self.H, self.W = 10, 10
        # For a single class and one image/slice:
        self.model_out = np.zeros((1, 1, self.H, self.W), dtype=np.int32)
        # Add a rectangle region (nonzero label) to simulate a segmentation:
        self.model_out[0, 0, 2:5, 2:5] = 1

    def test_call_valid(self):
        """Test that __call__ produces a valid AiCOCO output with annotations and objects."""
        output = self.strategy(
            model_out=self.model_out,
            images=[self.image],
            categories=self.categories_dict,
        ).model_dump()

        # Check that the basic AiCOCO fields exist.
        for key in ["images", "categories", "regressions", "meta", "annotations", "objects"]:
            self.assertIn(key, output)

        # Expect at least one annotation (for the drawn segmentation region).
        self.assertGreater(len(output["annotations"]), 0)
        # Since we have one label, one object should be created.
        self.assertEqual(len(output["objects"]), 1)

    def test_validate_model_output_valid(self):
        """Test that a valid segmentation output passes validation without error."""
        try:
            self.strategy.validate_model_output(self.model_out, self.categories_dict)
        except Exception as e:
            self.fail(f"validate_model_output() raised an exception unexpectedly: {e}")

    def test_validate_model_output_invalid_type(self):
        """Test that passing a non-numpy array raises a TypeError."""
        with self.assertRaises(TypeError):
            self.strategy.validate_model_output("not an array", self.categories_dict)

    def test_validate_model_output_wrong_length(self):
        """Test that a model output with mismatched number of classes raises a ValueError."""
        # Create an array with 2 classes while categories list has only 1 entry.
        invalid_model_out = np.zeros((2, 1, self.H, self.W), dtype=np.int32)
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(invalid_model_out, self.categories_dict)

    def test_validate_model_output_wrong_ndim(self):
        """Test that an array with invalid dimensions raises a ValueError."""
        # Create a 2D array (should be 3D or 4D).
        invalid_model_out = np.zeros((self.H, self.W), dtype=np.int32)
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(invalid_model_out, self.categories_dict)

    def test_validate_model_output_nonint(self):
        """Test that a float array (non-integer values) raises a ValueError."""
        invalid_model_out = np.zeros((1, 1, self.H, self.W), dtype=np.float32)
        invalid_model_out[0, 0, 2:5, 2:5] = 1.5  # non-integer values
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(invalid_model_out, self.categories_dict)

    def test_call_no_segmentation(self):
        """Test the strategy when the segmentation mask has no nonzero regions."""
        # Create a mask with all zeros.
        model_out_no_seg = np.zeros((1, 1, self.H, self.W), dtype=np.int32)
        output = self.strategy(
            model_out=model_out_no_seg,
            images=[self.image],
            categories=self.categories_dict,
        ).model_dump()

        # With no regions found, annotations and objects should be empty.
        self.assertEqual(len(output["annotations"]), 0)
        self.assertEqual(len(output["objects"]), 0)


@unittest.skip
class TestAiCOCOClassificationOutputStrategy(unittest.TestCase):
    def setUp(self):
        global global_counter
        global_counter = 0
        self.strategy = AiCOCOClassificationOutputStrategy()

    def get_categories_dict(self) -> List[Dict[str, Any]]:
        # Create two dummy category definitions.
        return [
            {"name": "class0", "supercategory_name": "group", "display": True},
            {"name": "class1", "supercategory_name": "group", "display": True},
        ]

    def test_validate_model_output_valid(self):
        # Valid model_out: 1D numpy array of int values with length equal to number of categories.
        model_out = np.array([1, 0], dtype=np.int32)
        categories = self.get_categories_dict()
        try:
            self.strategy.validate_model_output(model_out, categories)
        except Exception as e:
            self.fail(f"validate_model_output() raised an unexpected exception: {e}")

    def test_validate_model_output_invalid_type(self):
        with self.assertRaises(TypeError):
            self.strategy.validate_model_output("not an array", self.get_categories_dict())

    def test_validate_model_output_wrong_length(self):
        # model_out length is 1 while categories list length is 2.
        model_out = np.array([1], dtype=np.int32)
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(model_out, self.get_categories_dict())

    def test_validate_model_output_wrong_ndim(self):
        # model_out is not 1D.
        model_out = np.array([[1, 0]], dtype=np.int32)
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(model_out, self.get_categories_dict())

    def test_validate_model_output_nonint(self):
        # Using a float array will trigger check_any_nonint (dummy_check_any_nonint returns True for non-integer dtype)
        model_out = np.array([1.5, 0.5], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(model_out, self.get_categories_dict())

    def test_call_valid_single_image(self):
        """
        For a single image, update_images_meta should update the image's `category_ids`
        directly.
        """
        # Create a valid classification output: 1D array with 2 classes.
        model_out = np.array([1, 0], dtype=np.int32)
        image = AiImage(
            file_name="img1.png", id="img1", index=0, category_ids=None, regressions=None
        )
        categories_dict = self.get_categories_dict()
        # Provide a dummy meta object so that update_images_meta can operate without error.
        meta = AiMeta(category_ids=None, regressions=None)
        output = self.strategy(
            model_out=model_out, images=[image], categories=categories_dict, meta=meta
        ).model_dump()

        # Check that the output contains the expected keys.
        for key in ["images", "categories", "regressions", "meta", "annotations", "objects"]:
            self.assertIn(key, output)
        # In single image case, the image's category_ids should be updated.
        # Since model_out[0] is 1, the first category's id (from generate_categories) should be appended.
        expected_category_id = output["categories"][0]["id"]
        self.assertEqual(image.category_ids, [expected_category_id])
        # The meta object remains unchanged.
        self.assertEqual(output["meta"], meta.model_dump())

    def test_call_valid_multiple_images(self):
        """
        For multiple images, update_images_meta should update the meta object's `category_ids`
        instead of modifying the images.
        """
        # Create a valid classification output: 1D array with 2 classes.
        model_out = np.array([0, 1], dtype=np.int32)
        image1 = AiImage(
            file_name="img1.png", id="img1", index=0, category_ids=None, regressions=None
        )
        image2 = AiImage(
            file_name="img2.png", id="img2", index=0, category_ids=None, regressions=None
        )
        categories_dict = self.get_categories_dict()
        meta = AiMeta(category_ids=None, regressions=None)
        output = self.strategy(
            model_out=model_out, images=[image1, image2], categories=categories_dict, meta=meta
        ).model_dump()

        # Check that the output contains the expected keys.
        for key in ["images", "categories", "regressions", "meta", "annotations", "objects"]:
            self.assertIn(key, output)
        # For multiple images, update_images_meta updates the meta's category_ids.
        # Since model_out[1] is 1, the second category's id should be added.
        expected_category_id = output["categories"][1]["id"]
        self.assertEqual(output["meta"]["category_ids"], [expected_category_id])
        # The images' category_ids should remain unchanged (i.e., still None).
        self.assertIsNone(image1.category_ids)
        self.assertIsNone(image2.category_ids)


class TestAiCOCODetectionOutputStrategy(unittest.TestCase):
    def setUp(self):
        # Instantiate the strategy.
        self.strategy = AiCOCODetectionOutputStrategy()

        # Create a dummy image (AiImage).
        self.image = {
            "file_name": "test.jpg",
            "id": "img_1",
            "index": 0,
            "category_ids": None,
            "regressions": None,
        }

        # Create dummy categories.
        # Each category dict should have at least 'name' and 'supercategory_name'.
        # A 'display' attribute is used by the strategy when iterating over predictions.
        self.categories = [
            {"name": "cat", "supercategory_name": "animal", "display": True},
            {"name": "dog", "supercategory_name": "animal", "display": True},
        ]

        # Create dummy regressions (empty list for most tests).
        self.regressions = []

        # A valid model output (with one detection) including only bbox and one-hot class.
        self.model_out = {
            "bbox_pred": [[10, 20, 30, 40]],
            "cls_pred": [[1, 0]],
        }

    def test_call_valid(self):
        """Test that __call__ returns a valid AiCOCO-like output."""
        result = self.strategy(
            model_out=self.model_out,
            images=[self.image],
            categories=self.categories,
            regressions=self.regressions,
        ).model_dump()

        # Check that the result contains the expected keys.
        for key in ["images", "categories", "regressions", "meta", "annotations", "objects"]:
            self.assertIn(key, result)
        # Expect one annotation and one object.
        self.assertEqual(len(result["annotations"]), 1)
        self.assertEqual(len(result["objects"]), 1)

        # Check the annotation's bbox and image_id.
        annot = result["annotations"][0]
        self.assertEqual(annot["bbox"], [[10, 20, 30, 40]])
        self.assertEqual(annot["image_id"], self.image["id"])

        # Check that the object has a non-empty category_ids (the predicted category).
        obj = result["objects"][0]
        self.assertTrue(obj["category_ids"])

    def test_validate_model_output_missing_bbox_pred(self):
        """Test that missing 'bbox_pred' raises a KeyError."""
        bad_model_out = {"cls_pred": [[1, 0]]}
        with self.assertRaises(KeyError):
            self.strategy.validate_model_output(bad_model_out, self.categories)

    def test_validate_model_output_missing_cls_pred(self):
        """Test that missing 'cls_pred' raises a KeyError."""
        bad_model_out = {"bbox_pred": [[10, 20, 30, 40]]}
        with self.assertRaises(KeyError):
            self.strategy.validate_model_output(bad_model_out, self.categories)

    def test_validate_model_output_length_mismatch(self):
        """Test that a length mismatch between bbox_pred and cls_pred raises ValueError."""
        bad_model_out = {
            "bbox_pred": [[10, 20, 30, 40], [50, 60, 70, 80]],
            "cls_pred": [[1, 0]],
        }
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(bad_model_out, self.categories)

    def test_validate_model_output_cls_pred_inner_length(self):
        """Test that an inner list in cls_pred with incorrect length raises ValueError."""
        bad_model_out = {
            "bbox_pred": [[10, 20, 30, 40]],
            "cls_pred": [[1]],  # Expected length is 2 (since there are 2 categories).
        }
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(bad_model_out, self.categories)

    def test_model_to_aicoco_with_confidence_and_regression(self):
        """Test __call__ when confidence_score and regression_value are provided."""
        # Create a model output with confidence and regression values.
        model_out = {
            "bbox_pred": [[15, 25, 35, 45]],
            "cls_pred": [[1, 0]],
            "confidence_score": [0.9],
            "regression_value": [[0.5, 0.7]],
        }
        # Provide a dummy regression entry.
        regressions = [{"name": "size", "superregression_name": None}]
        result = self.strategy(
            model_out=model_out,
            images=[self.image],
            categories=self.categories,
            regressions=regressions,
        )

        # Check that an annotation and an object are created.
        self.assertEqual(len(result.annotations), 1)
        self.assertEqual(len(result.objects), 1)

        obj = result.objects[0]
        # The object should now have a 'confidence' attribute.
        self.assertTrue(hasattr(obj, "confidence"))
        self.assertEqual(obj.confidence, 0.9)

        # The object should have a regression item.
        self.assertTrue(obj.regressions)

    def test_generate_annotations_objects_no_detection(self):
        """Test that if no category is predicted (all zeros), no annotation/object is generated."""
        model_out = {
            "bbox_pred": [[10, 20, 30, 40]],
            "cls_pred": [[0, 0]],  # No detection.
        }
        result = self.strategy(
            model_out=model_out,
            images=[self.image],
            categories=self.categories,
            regressions=self.regressions,
        ).model_dump()

        # No annotations or objects should be added.
        self.assertEqual(len(result["annotations"]), 0)
        self.assertEqual(len(result["objects"]), 0)


if __name__ == "__main__":
    unittest.main()
