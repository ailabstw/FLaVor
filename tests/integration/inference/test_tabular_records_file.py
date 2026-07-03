import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from flavor.serve.inference.data_models.api import AiCOCOTabularOutputDataModel
from flavor.serve.inference.strategies.aicoco_strategy import (
    AiCOCOTabularClassificationOutputStrategy,
)


def test_tabular_output_model_uses_records_file_instead_of_records(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("{}\n", encoding="utf-8")

    result = AiCOCOTabularOutputDataModel.model_validate(
        {
            "tables": [{"id": "table_1", "file_name": "infer.csv"}],
            "categories": [{"id": "0", "name": "Normal", "supercategory_id": None}],
            "regressions": [],
            "records_file": {
                "format": "jsonl",
                "path": str(records_path),
                "rows": 1,
                "bytes": records_path.stat().st_size,
            },
            "meta": {"window_size": 1},
        }
    )

    assert result.records_file.format == "jsonl"
    assert result.records_file.rows == 1
    assert result.records_file.path == str(records_path)

    with pytest.raises(ValidationError):
        AiCOCOTabularOutputDataModel.model_validate(
            {
                "tables": [{"id": "table_1", "file_name": "infer.csv"}],
                "categories": [],
                "regressions": [],
                "records": [],
                "meta": {"window_size": 1},
            }
        )


def test_tabular_classification_strategy_writes_records_jsonl(tmp_path):
    strategy = AiCOCOTabularClassificationOutputStrategy()

    result = strategy(
        model_out=np.array([[1, 0], [0, 1], [1, 0]]),
        tables=[{"id": "table_1", "file_name": "infer.csv"}],
        dataframes=[pd.DataFrame({"amount": [10, 20, 30]})],
        categories=[{"id": "0", "name": "Normal"}, {"id": "1", "name": "Fraud"}],
        meta={"window_size": 1},
        records_output_dir=tmp_path,
    )

    records_path = tmp_path / result.records_file.path.split("/")[-1]
    assert result.records_file.format == "jsonl"
    assert result.records_file.rows == 3
    assert result.records_file.bytes == records_path.stat().st_size

    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["table_id"] for record in records] == ["table_1"] * 3
    assert [record["row_indexes"] for record in records] == [[0], [1], [2]]
    normal_id, fraud_id = [category.id for category in result.categories]
    assert [record["category_ids"] for record in records] == [[normal_id], [fraud_id], [normal_id]]
