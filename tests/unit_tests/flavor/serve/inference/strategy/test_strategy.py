import unittest
from typing import Any, Dict, List

import numpy as np
import pandas as pd

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
    AiCOCOTabularFormat,
    AiCOCOTabularRegressionOutputStrategy,
    BaseAiCOCOOutputStrategy,
    BaseAiCOCOTabularOutputStrategy,
)


def create_dummy_image(
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


def get_dummy_categories_dict() -> List[Dict[str, Any]]:
    """Helper to return a sample categories list (dict format)."""
    return [
        {"name": "cat", "supercategory_name": "animal", "display": True},
        {"name": "dog", "supercategory_name": "animal", "display": True},
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
        input_categories = [
            {"name": "cat", "supercategory_name": "animal"},
            {"name": "dog", "supercategory_name": "animal"},
            {"name": "car", "supercategory_name": "vehicle"},
        ]
        result = self.strategy.generate_categories(input_categories)
        # Input 3 items should result in 3 processed items + 2 additional supercategories
        self.assertEqual(len(result), 5)

        # Ensure "supercategory_name" is no longer present in the returned objects
        for cat in result:
            self.assertFalse(hasattr(cat, "supercategory_name"))

        # Check the extra supercategories created (with supercategory_id set to None)
        supercategories = [cat for cat in result if cat.supercategory_id is None]
        self.assertEqual(len(supercategories), 2)
        expected_names = sorted(["animal", "vehicle"])
        actual_names = sorted([cat.name for cat in supercategories])
        self.assertEqual(actual_names, expected_names)

        # Check that each normal category's supercategory_id points to a valid supercategory
        valid_super_ids = {cat.id for cat in supercategories}
        for cat in result:
            if cat.supercategory_id is not None:
                self.assertIn(cat.supercategory_id, valid_super_ids)

    def test_generate_regressions(self):
        input_regressions = [
            {"name": "reg1", "superregression_name": "sup1"},
            {"name": "reg2", "superregression_name": "sup1"},
            {"name": "reg3", "superregression_name": "sup2"},
        ]
        result = self.strategy.generate_regressions(input_regressions)
        # Expect 3 input items + 2 additional superregressions
        self.assertEqual(len(result), 5)

        # Ensure "superregression_name" is no longer present in the returned objects
        for reg in result:
            self.assertFalse(hasattr(reg, "superregression_name"))

        superregs = [reg for reg in result if reg.superregression_id is None]
        self.assertEqual(len(superregs), 2)
        expected_sup_names = sorted(["sup1", "sup2"])
        actual_sup_names = sorted([reg.name for reg in superregs])
        self.assertEqual(actual_sup_names, expected_sup_names)

        valid_super_ids = {reg.id for reg in superregs}
        for reg in result:
            if reg.superregression_id is not None:
                self.assertIn(reg.superregression_id, valid_super_ids)

    def test_prepare_aicoco(self):
        image = create_dummy_image(id="img1", file_name="image1.jpg")
        images = [image]
        input_categories = [{"name": "cat", "supercategory_name": "animal"}]
        input_regressions = [{"name": "reg1", "superregression_name": "sup1"}]
        meta = {"category_ids": ["cat1"], "regressions": [{"regression_id": "r1", "value": 0.5}]}

        aicoco_ref = self.strategy.prepare_aicoco(
            images=images,
            categories=input_categories,
            regressions=input_regressions,
            meta=meta,
        )
        self.assertIsInstance(aicoco_ref, AiCOCORef)
        self.assertEqual(aicoco_ref.images, images)
        self.assertEqual(aicoco_ref.meta, AiMeta(**meta))
        # One input category results in 1 processed item + 1 extra supercategory
        self.assertEqual(len(aicoco_ref.categories), 2)
        # One input regression results in 1 processed item + 1 extra superregression
        self.assertEqual(len(aicoco_ref.regressions), 2)


class TestAiCOCOSegmentationOutputStrategy(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = AiCOCOSegmentationOutputStrategy()
        self.image = create_dummy_image(id="img1", file_name="img1.png")
        self.categories_dict = [{"name": "object", "supercategory_name": "all", "display": True}]
        self.H, self.W = 10, 10
        # Create a 4D segmentation mask (1 class, 1 slice)
        self.model_out = np.zeros((1, 1, self.H, self.W), dtype=np.int32)
        self.model_out[0, 0, 2:5, 2:5] = 1

    def test_call_valid(self):
        """Test correct segmentation output that should produce annotations and objects"""
        result = self.strategy(
            model_out=self.model_out,
            images=[self.image],
            categories=self.categories_dict,
        )

        for attr in ["images", "categories", "regressions", "meta", "annotations", "objects"]:
            self.assertTrue(hasattr(result, attr))
        self.assertGreater(len(result.annotations), 0)
        self.assertEqual(len(result.objects), 1)

    def test_validate_model_output_valid(self):
        try:
            self.strategy.validate_model_output(self.model_out, self.categories_dict)
        except Exception as e:
            self.fail(f"validate_model_output() should not raise an exception: {e}")

    def test_validate_model_output_invalid_type(self):
        with self.assertRaises(TypeError):
            self.strategy.validate_model_output("not an array", self.categories_dict)

    def test_validate_model_output_wrong_length(self):
        invalid_out = np.zeros((2, 1, self.H, self.W), dtype=np.int32)
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(invalid_out, self.categories_dict)

    def test_validate_model_output_wrong_ndim(self):
        invalid_out = np.zeros((self.H, self.W), dtype=np.int32)
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(invalid_out, self.categories_dict)

    def test_validate_model_output_nonint(self):
        invalid_out = np.zeros((1, 1, self.H, self.W), dtype=np.float32)
        invalid_out[0, 0, 2:5, 2:5] = 1.5
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(invalid_out, self.categories_dict)

    def test_call_no_segmentation(self):
        model_out_empty = np.zeros((1, 1, self.H, self.W), dtype=np.int32)
        result = self.strategy(
            model_out=model_out_empty,
            images=[self.image],
            categories=self.categories_dict,
        )
        self.assertEqual(len(result.annotations), 0)
        self.assertEqual(len(result.objects), 0)


class TestAiCOCOClassificationOutputStrategy(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = AiCOCOClassificationOutputStrategy()

    def get_categories(self) -> List[Dict[str, Any]]:
        return [
            {"name": "class0", "supercategory_name": "group", "display": True},
            {"name": "class1", "supercategory_name": "group", "display": True},
        ]

    def test_validate_model_output_valid(self):
        model_out = np.array([1, 0], dtype=np.int32)
        categories = self.get_categories()
        try:
            self.strategy.validate_model_output(model_out, categories)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")

    def test_validate_model_output_invalid_type(self):
        with self.assertRaises(TypeError):
            self.strategy.validate_model_output("not an array", self.get_categories())

    def test_validate_model_output_wrong_length(self):
        model_out = np.array([1], dtype=np.int32)
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(model_out, self.get_categories())

    def test_validate_model_output_wrong_ndim(self):
        model_out = np.array([[1, 0]], dtype=np.int32)
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(model_out, self.get_categories())

    def test_validate_model_output_nonint(self):
        model_out = np.array([1.5, 0.5], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(model_out, self.get_categories())

    def test_call_valid_single_image(self):
        model_out = np.array([1, 0], dtype=np.int32)
        image = create_dummy_image(id="img1", file_name="img1.png")
        categories = self.get_categories()
        meta = get_dummy_meta()
        result = self.strategy(
            model_out=model_out, images=[image], categories=categories, meta=meta
        )
        expected_cat_id = result.categories[0].id
        self.assertEqual(image.category_ids, [expected_cat_id])
        self.assertEqual(result.meta, meta)

    def test_call_valid_multiple_images(self):
        model_out = np.array([0, 1], dtype=np.int32)
        image1 = create_dummy_image(id="img1", file_name="img1.png")
        image2 = create_dummy_image(id="img2", file_name="img2.png")
        categories = self.get_categories()
        meta = get_dummy_meta()
        result = self.strategy(
            model_out=model_out, images=[image1, image2], categories=categories, meta=meta
        )
        expected_cat_id = result.categories[1].id
        self.assertEqual(result.meta.category_ids, [expected_cat_id])
        self.assertIsNone(image1.category_ids)
        self.assertIsNone(image2.category_ids)


class TestAiCOCODetectionOutputStrategy(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = AiCOCODetectionOutputStrategy()
        self.image = {
            "file_name": "test.jpg",
            "id": "img_1",
            "index": 0,
            "category_ids": None,
            "regressions": None,
        }
        self.categories = [
            {"name": "cat", "supercategory_name": "animal", "display": True},
            {"name": "dog", "supercategory_name": "animal", "display": True},
        ]
        self.regressions = []
        self.model_out = {"bbox_pred": [[10, 20, 30, 40]], "cls_pred": [[1, 0]]}

    def test_call_valid(self):
        result = self.strategy(
            model_out=self.model_out,
            images=[self.image],
            categories=self.categories,
            regressions=self.regressions,
        )

        for attr in ["images", "categories", "regressions", "meta", "annotations", "objects"]:
            self.assertTrue(hasattr(result, attr))
        self.assertEqual(len(result.annotations), 1)
        self.assertEqual(len(result.objects), 1)
        annot = result.annotations[0]
        self.assertEqual(annot.bbox, [[10, 20, 30, 40]])
        self.assertEqual(annot.image_id, self.image["id"])
        obj = result.objects[0]
        self.assertTrue(obj.category_ids)

    def test_validate_model_output_missing_bbox_pred(self):
        bad_model_out = {"cls_pred": [[1, 0]]}
        with self.assertRaises(KeyError):
            self.strategy.validate_model_output(bad_model_out, self.categories)

    def test_validate_model_output_missing_cls_pred(self):
        bad_model_out = {"bbox_pred": [[10, 20, 30, 40]]}
        with self.assertRaises(KeyError):
            self.strategy.validate_model_output(bad_model_out, self.categories)

    def test_validate_model_output_length_mismatch(self):
        bad_model_out = {
            "bbox_pred": [[10, 20, 30, 40], [50, 60, 70, 80]],
            "cls_pred": [[1, 0]],
        }
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(bad_model_out, self.categories)

    def test_validate_model_output_cls_pred_inner_length(self):
        bad_model_out = {"bbox_pred": [[10, 20, 30, 40]], "cls_pred": [[1]]}
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(bad_model_out, self.categories)

    def test_model_to_aicoco_with_confidence_and_regression(self):
        model_out = {
            "bbox_pred": [[15, 25, 35, 45]],
            "cls_pred": [[1, 0]],
            "confidence_score": [0.9],
            "regression_value": [[0.5, 0.7]],
        }
        regressions = [{"name": "size", "superregression_name": None}]
        result = self.strategy(
            model_out=model_out,
            images=[self.image],
            categories=self.categories,
            regressions=regressions,
        )
        self.assertEqual(len(result.annotations), 1)
        self.assertEqual(len(result.objects), 1)
        obj = result.objects[0]
        self.assertTrue(hasattr(obj, "confidence"))
        self.assertEqual(obj.confidence, 0.9)
        self.assertIsNotNone(obj.regressions)

    def test_generate_annotations_objects_no_detection(self):
        model_out = {"bbox_pred": [[10, 20, 30, 40]], "cls_pred": [[0, 0]]}
        result = self.strategy(
            model_out=model_out,
            images=[self.image],
            categories=self.categories,
            regressions=self.regressions,
        )
        self.assertEqual(len(result.annotations), 0)
        self.assertEqual(len(result.objects), 0)


class TestAiCOCORegressionOutputStrategy(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = AiCOCORegressionOutputStrategy()
        self.image = create_dummy_image(id="img_1", file_name="test.jpg")
        self.regressions_dict = [{"name": "size", "superregression_name": None}]

    def test_validate_model_output_valid_1d(self):
        model_out = np.array([0.5])
        try:
            self.strategy.validate_model_output(model_out)
        except Exception as e:
            self.fail(f"Unexpected exception for valid 1D array: {e}")

    def test_validate_model_output_valid_4d(self):
        model_out = np.zeros((1, 1, 1, 1))
        try:
            self.strategy.validate_model_output(model_out)
        except Exception as e:
            self.fail(f"Unexpected exception for valid 4D array: {e}")

    def test_validate_model_output_wrong_type(self):
        with self.assertRaises(TypeError):
            self.strategy.validate_model_output([0.5])

    def test_validate_model_output_wrong_ndim(self):
        model_out = np.zeros((2, 2))
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(model_out)

    def test_call_single_image(self):
        model_out = np.array([0.75])
        result = self.strategy(
            model_out=model_out,
            images=[self.image],
            regressions=self.regressions_dict,
        )
        updated_image = result.images[0]
        self.assertIsNotNone(updated_image.regressions)
        self.assertEqual(len(updated_image.regressions), 1)
        reg_item = updated_image.regressions[0]
        self.assertEqual(reg_item.value, 0.75)
        meta_regressions = (
            result.meta.get("regressions")
            if isinstance(result.meta, dict)
            else result.meta.regressions
        )
        self.assertTrue(meta_regressions in (None, []))

    def test_call_multiple_images(self):
        model_out = np.array([0.85])
        image1 = create_dummy_image(id="img1", file_name="img1.jpg")
        image2 = create_dummy_image(id="img2", file_name="img2.jpg")
        result = self.strategy(
            model_out=model_out,
            images=[image1, image2],
            regressions=self.regressions_dict,
        )
        meta = result.meta
        reg_list = meta.get("regressions") if isinstance(meta, dict) else meta.regressions
        self.assertIsNotNone(reg_list)
        self.assertEqual(len(reg_list), 1)
        reg_item = reg_list[0]
        self.assertEqual(reg_item.value, 0.85)
        for img in result.images:
            self.assertIsNone(img.regressions)


class DummyTabularOutputStrategy(BaseAiCOCOTabularOutputStrategy):
    def __call__(self, *args, **kwargs):
        pass

    def model_to_aicoco(self, aicoco_ref: Any, model_out: Any, **kwargs) -> Any:
        return aicoco_ref


class TestBaseAiCOCOTabularOutputStrategy(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = DummyTabularOutputStrategy()

    def test_generate_records(self):
        df = pd.DataFrame({"a": range(5)})
        meta = {"window_size": 2}
        table = AiTable(id="table1", file_name="dummy.csv")
        records = self.strategy.generate_records([df], [table], meta)

        self.assertEqual(len(records), 2)
        expected_indexes = [[0, 1], [2, 3]]
        for rec, exp in zip(records, expected_indexes):
            self.assertEqual(rec.table_id, "table1")
            self.assertEqual(rec.row_indexes, exp)

    def test_prepare_aicoco(self):
        tables_input = [{"id": "table1", "file_name": "dummy.csv"}]
        meta = {"window_size": 2}
        df = pd.DataFrame({"a": range(4)})
        dataframes = [df]
        categories_input = [{"name": "Category1"}]
        regressions_input = [{"name": "Regression1"}]
        aicoco_output = self.strategy.prepare_aicoco(
            tables=tables_input,
            meta=meta,
            dataframes=dataframes,
            categories=categories_input,
            regressions=regressions_input,
        )
        self.assertIsInstance(aicoco_output, AiCOCOTabularFormat)
        self.assertEqual(len(aicoco_output.tables), 1)
        self.assertEqual(aicoco_output.tables[0].id, "table1")
        self.assertEqual(len(aicoco_output.records), 2)
        expected_indexes = [[0, 1], [2, 3]]
        for rec, exp in zip(aicoco_output.records, expected_indexes):
            self.assertEqual(rec.table_id, "table1")
            self.assertEqual(rec.row_indexes, exp)
        self.assertEqual(len(aicoco_output.categories), 1)
        self.assertEqual(aicoco_output.categories[0].name, "Category1")
        self.assertEqual(len(aicoco_output.regressions), 1)
        self.assertEqual(aicoco_output.regressions[0].name, "Regression1")
        self.assertEqual(aicoco_output.meta.window_size, 2)


class TestAiCOCOTabularClassificationOutputStrategy(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = AiCOCOTabularClassificationOutputStrategy()
        self.categories = [{"name": "Category 1"}, {"name": "Category 2"}]

    def test_validate_model_output_valid(self):
        model_out = np.array([[1, 0], [0, 1]])
        try:
            self.strategy.validate_model_output(model_out, self.categories)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")

    def test_validate_model_output_invalid_type(self):
        model_out = [[1, 0], [0, 1]]
        with self.assertRaises(TypeError) as context:
            self.strategy.validate_model_output(model_out, self.categories)
        self.assertIn("np.ndarray", str(context.exception))

    def test_validate_model_output_invalid_dimension(self):
        model_out = np.array([[[1]]])
        with self.assertRaises(ValueError) as context:
            self.strategy.validate_model_output(model_out, self.categories)
        self.assertIn("dimension must be <=", str(context.exception))

    def test_validate_model_output_mismatched_last_dim(self):
        model_out = np.array([[1, 0, 1]])
        with self.assertRaises(ValueError) as context:
            self.strategy.validate_model_output(model_out, self.categories)
        self.assertIn("number of classes", str(context.exception))

    def test_validate_model_output_non_binary(self):
        model_out = np.array([[2, 0]])
        with self.assertRaises(ValueError) as context:
            self.strategy.validate_model_output(model_out, self.categories)
        self.assertIn("contains elements other than 0 and 1", str(context.exception))

    def test_model_to_aicoco(self):
        tables = [{"id": "table1", "file_name": "dummy.csv"}]
        dataframes = [pd.DataFrame({"a": [1]})]
        meta = {"window_size": 1}
        aicoco_ref = self.strategy.prepare_aicoco(tables, meta, dataframes, self.categories)
        model_out = np.array([[1, 0]])
        out_ref = self.strategy.model_to_aicoco(aicoco_ref, model_out)
        self.assertIsInstance(out_ref, AiCOCOTabularFormat)
        self.assertEqual(out_ref.records[0].category_ids[0], out_ref.categories[0].id)

    def test_call_success(self):
        dataframes = [pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]})]
        tables = [
            {"id": "table1", "file_name": "table1.csv"},
            {"id": "table2", "file_name": "table2.csv"},
        ]
        meta = {"window_size": 1}
        model_out = np.array([[1, 0], [0, 1]])
        aicoco_out = self.strategy(
            model_out=model_out,
            tables=tables,
            dataframes=dataframes,
            categories=self.categories,
            meta=meta,
        )
        records = aicoco_out.records
        categories = aicoco_out.categories

        self.assertEqual(len(records), 2)
        self.assertEqual(categories[0].name, self.categories[0]["name"])
        self.assertEqual(categories[1].name, self.categories[1]["name"])
        self.assertEqual(records[0].category_ids[0], categories[0].id)
        self.assertEqual(records[1].category_ids[0], categories[1].id)

    def test_call_mismatched_records(self):
        dataframes = [pd.DataFrame({"a": [1]})]
        tables = [{"id": "table1", "file_name": "table1.csv"}]
        meta = {"window_size": 2}
        model_out = np.array([[1, 0], [0, 1]])
        with self.assertRaises(ValueError) as context:
            _ = self.strategy(
                model_out=model_out,
                tables=tables,
                dataframes=dataframes,
                categories=self.categories,
                meta=meta,
            )
        self.assertIn("number of records is not matched", str(context.exception))


class TestAiCOCOTabularRegressionOutputStrategy(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = AiCOCOTabularRegressionOutputStrategy()
        self.tables = [
            {"id": "table1", "file_name": "table1.csv"},
            {"id": "table2", "file_name": "table2.csv"},
        ]
        self.dataframes = [pd.DataFrame({"col": [1]}), pd.DataFrame({"col": [2]})]
        self.meta = {"window_size": 1}
        self.regressions = [
            {"name": "reg1"},
            {"name": "reg2"},
        ]

    def test_validate_model_output_valid(self):
        model_out = np.array([[1.0, 2.0], [3.0, 4.0]])
        try:
            self.strategy.validate_model_output(model_out)
        except Exception as e:
            self.fail(f"validate_model_output() raised an unexpected exception: {e}")

    def test_validate_model_output_wrong_type(self):
        with self.assertRaises(TypeError):
            self.strategy.validate_model_output([[1.0, 2.0]])

    def test_validate_model_output_too_high_dim(self):
        model_out = np.zeros((1, 1, 1))
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(model_out)

    def test_validate_model_output_inf_values(self):
        model_out = np.array([[np.inf]])
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(model_out)

    def test_call_single_regression(self):
        model_out = np.array([[0.5], [0.7]])  # 2 records × 1 regression
        result = self.strategy(
            model_out=model_out,
            tables=self.tables,
            dataframes=self.dataframes,
            regressions=[{"name": "reg1"}],
            meta=self.meta,
        )
        self.assertIsInstance(result, AiCOCOTabularFormat)
        self.assertEqual(len(result.records), 2)
        for record in result.records:
            self.assertEqual(len(record.regressions), 1)
            self.assertIsInstance(record.regressions[0], AiRegressionItem)

    def test_call_multiple_regression(self):
        model_out = np.array([[1.1, 2.2], [3.3, 4.4]])  # 2 records × 2 regressions
        result = self.strategy(
            model_out=model_out,
            tables=self.tables,
            dataframes=self.dataframes,
            regressions=self.regressions,
            meta=self.meta,
        )
        self.assertEqual(len(result.records), 2)
        for record in result.records:
            self.assertEqual(len(record.regressions), 2)
            for reg in record.regressions:
                self.assertIsInstance(reg, AiRegressionItem)

    def test_model_output_length_mismatch(self):
        model_out = np.array([[1.0], [2.0], [3.0]])  # 3 outputs but only 2 records
        with self.assertRaises(ValueError) as ctx:
            self.strategy(
                model_out=model_out,
                tables=self.tables,
                dataframes=self.dataframes,
                regressions=[{"name": "reg1"}],
                meta=self.meta,
            )
        self.assertIn("number of records is not matched", str(ctx.exception))


class TestAiCOCOHybridClassificationOutputStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = AiCOCOHybridClassificationOutputStrategy()
        self.images = [
            create_dummy_image(id="img1", file_name="img1.jpg"),
            create_dummy_image(id="img2", file_name="img2.jpg"),
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
        result = self.strategy(
            model_out=self.model_out,
            images=self.images,
            tables=self.tables,
            categories=self.categories,
        )
        self.assertTrue(hasattr(result, "images"))
        self.assertTrue(hasattr(result, "categories"))
        self.assertTrue(hasattr(result, "tables"))
        self.assertEqual(len(result.tables), 2)
        self.assertIsInstance(result.tables[0], AiTable)
        self.assertEqual(result.tables[0].file_name, "table1.csv")

    def test_model_to_aicoco_adds_tables(self):
        aicoco_ref = self.strategy.prepare_aicoco(images=self.images, categories=self.categories)
        result = self.strategy.model_to_aicoco(
            aicoco_ref=aicoco_ref,
            model_out=self.model_out,
            tables=self.tables,
        )
        self.assertTrue(hasattr(result, "tables"))
        self.assertEqual(len(result.tables), 2)
        for table in result.tables:
            self.assertIsInstance(table, AiTable)

    def test_validate_model_output_invalid_type(self):
        bad_output = "not an array"
        with self.assertRaises(TypeError):
            self.strategy.validate_model_output(bad_output, self.categories)

    def test_validate_model_output_wrong_length(self):
        wrong_length_output = np.array([1], dtype=np.int32)
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(wrong_length_output, self.categories)

    def test_validate_model_output_wrong_dtype(self):
        bad_output = np.array([0.5, 1.5], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(bad_output, self.categories)


class TestAiCOCOHybridRegressionOutputStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = AiCOCOHybridRegressionOutputStrategy()
        self.images = [
            create_dummy_image(id="img1", file_name="img1.jpg"),
        ]
        self.model_out = np.array([1.0], dtype=np.float32)
        self.regressions = [
            {"name": "reg1"},
        ]
        self.tables = [
            {"id": "table1", "file_name": "table1.csv"},
        ]

    def test_call_valid(self):
        """Test that __call__ returns a result with images, regressions, and tables"""
        result = self.strategy(
            model_out=self.model_out,
            images=self.images,
            tables=self.tables,
            regressions=self.regressions,
        )
        self.assertTrue(hasattr(result, "images"))
        self.assertTrue(hasattr(result, "regressions"))
        self.assertTrue(hasattr(result, "tables"))
        self.assertEqual(len(result.tables), 1)
        self.assertIsInstance(result.tables[0], AiTable)
        self.assertEqual(result.tables[0].file_name, "table1.csv")

    def test_model_to_aicoco_adds_tables(self):
        """Test that model_to_aicoco correctly adds tables and regressions"""
        aicoco_ref = self.strategy.prepare_aicoco(images=self.images, regressions=self.regressions)
        result = self.strategy.model_to_aicoco(
            aicoco_ref=aicoco_ref,
            model_out=self.model_out,
            tables=self.tables,
        )
        self.assertTrue(hasattr(result, "tables"))
        self.assertEqual(len(result.tables), 1)
        for table in result.tables:
            self.assertIsInstance(table, AiTable)

    def test_validate_model_output_invalid_type(self):
        """Test that validate_model_output raises TypeError for non-ndarray input"""
        bad_output = "not an array"
        with self.assertRaises(TypeError):
            self.strategy.validate_model_output(bad_output)

    def test_validate_model_output_wrong_length(self):
        """Test that validate_model_output raises ValueError for incorrect output length"""
        wrong_length_output = np.array([[1.0], [2.0]], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.strategy.validate_model_output(wrong_length_output)


if __name__ == "__main__":
    unittest.main()
