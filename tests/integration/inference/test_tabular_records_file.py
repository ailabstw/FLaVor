import json

import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, ValidationError

from flavor.serve.invocations.infer_invocation import InferInvocationAPP
from flavor.serve.inference.data_models.api import AiCOCOTabularOutputDataModel
from flavor.serve.inference.inference_models import BaseAiCOCOTabularInferenceModel
from flavor.serve.inference.strategies.aicoco_strategy import (
    AiCOCOTabularClassificationOutputStrategy,
)


class LegacyTabularInferenceModel(BaseAiCOCOTabularInferenceModel):
    def define_inference_network(self):
        return lambda dataframes: np.array([[1, 0]])

    def set_categories(self):
        return [{"name": "Normal"}, {"name": "Fraud"}]

    def set_regressions(self):
        return None

    def data_reader(self, files):
        return [pd.DataFrame({"amount": [10]})]

    def output_formatter(self, model_out, tables, dataframes, meta, categories=None):
        return self.formatter(
            model_out=model_out,
            tables=tables,
            dataframes=dataframes,
            categories=categories,
            meta=meta,
        )


class EmptyInputDataModel(BaseModel):
    pass


def test_tabular_output_model_uses_records_artifact_href():
    result = AiCOCOTabularOutputDataModel.model_validate(
        {
            "tables": [{"id": "table_1", "file_name": "infer.csv"}],
            "categories": [{"id": "0", "name": "Normal", "supercategory_id": None}],
            "regressions": [],
            "records": {
                "format": "jsonl",
                "href": "/invocations/artifacts/records_abc.jsonl",
                "rows": 1,
                "bytes": 3,
            },
            "meta": {"window_size": 1},
        }
    )

    assert result.records.format == "jsonl"
    assert result.records.href == "/invocations/artifacts/records_abc.jsonl"
    assert result.records.rows == 1
    assert result.records.bytes == 3
    assert "expires_at" not in result.records.model_dump()

    with pytest.raises(ValidationError):
        AiCOCOTabularOutputDataModel.model_validate(
            {
                "tables": [{"id": "table_1", "file_name": "infer.csv"}],
                "categories": [{"id": "0", "name": "Normal", "supercategory_id": None}],
                "regressions": [],
                "records": {
                    "format": "jsonl",
                    "href": "/invocations/artifacts/records_abc.jsonl",
                    "rows": 1,
                    "bytes": 3,
                    "expires_at": None,
                },
                "meta": {"window_size": 1},
            }
        )

    with pytest.raises(ValidationError):
        AiCOCOTabularOutputDataModel.model_validate(
            {
                "tables": [{"id": "table_1", "file_name": "infer.csv"}],
                "categories": [],
                "regressions": [],
                "records_file": {
                    "format": "jsonl",
                    "path": "/tmp/records.jsonl",
                    "rows": 1,
                    "bytes": 3,
                },
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

    artifact_name = result.records.href.rsplit("/", 1)[-1]
    records_path = tmp_path / artifact_name
    assert result.records.format == "jsonl"
    assert result.records.href == f"/invocations/artifacts/{artifact_name}"
    assert result.records.rows == 3
    assert result.records.bytes == records_path.stat().st_size
    assert "expires_at" not in result.records.model_dump()

    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["table_id"] for record in records] == ["table_1"] * 3
    assert [record["row_indexes"] for record in records] == [[0], [1], [2]]
    normal_id, fraud_id = [category.id for category in result.categories]
    assert [record["category_ids"] for record in records] == [[normal_id], [fraud_id], [normal_id]]


@pytest.mark.asyncio
async def test_invocations_serves_tabular_records_artifact_href(tmp_path):
    artifact_name = "records_0123456789ABCDEFGHIJK.jsonl"
    records_path = tmp_path / artifact_name
    content = (
        '{"id":"record_1","table_id":"table_1","row_indexes":[0],'
        '"category_ids":["normal"],"regressions":null}\n'
    )
    records_path.write_text(content, encoding="utf-8")

    def infer_function():
        return {
            "tables": [{"id": "table_1", "file_name": "infer.csv"}],
            "categories": [{"id": "normal", "name": "Normal", "supercategory_id": None}],
            "regressions": [],
            "records": {
                "format": "jsonl",
                "href": f"/invocations/artifacts/{artifact_name}",
                "rows": 1,
                "bytes": records_path.stat().st_size,
            },
            "meta": {"window_size": 1},
        }

    app = InferInvocationAPP(
        infer_function,
        EmptyInputDataModel,
        AiCOCOTabularOutputDataModel,
        records_output_dir=tmp_path,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app.app), base_url="http://test"
    ) as client:
        response = await client.post("/invocations", data={})
        assert response.status_code == 200
        href = response.json()["records"]["href"]

        records_response = await client.get(href)

    assert records_response.status_code == 200
    assert records_response.headers["content-type"].startswith("application/x-ndjson")
    assert records_response.text == content


@pytest.mark.asyncio
async def test_invocations_writes_and_serves_generated_records_artifact(tmp_path):
    strategy = AiCOCOTabularClassificationOutputStrategy()

    def infer_function(**kwargs):
        return strategy(
            model_out=np.array([[1, 0], [0, 1]]),
            tables=[{"id": "table_1", "file_name": "infer.csv"}],
            dataframes=[pd.DataFrame({"amount": [10, 20]})],
            categories=[{"id": "0", "name": "Normal"}, {"id": "1", "name": "Fraud"}],
            meta={"window_size": 1},
            **kwargs,
        )

    app = InferInvocationAPP(
        infer_function,
        EmptyInputDataModel,
        AiCOCOTabularOutputDataModel,
        records_output_dir=tmp_path,
        records_href_prefix="/custom-artifacts",
    )

    async with AsyncClient(
        transport=ASGITransport(app=app.app), base_url="http://test"
    ) as client:
        response = await client.post("/invocations", data={})
        assert response.status_code == 200
        href = response.json()["records"]["href"]
        assert href.startswith("/custom-artifacts/records_")

        artifact_name = href.rsplit("/", 1)[-1]
        assert (tmp_path / artifact_name).is_file()

        records_response = await client.get(href)

    assert records_response.status_code == 200
    records = [json.loads(line) for line in records_response.text.splitlines()]
    assert [record["row_indexes"] for record in records] == [[0], [1]]


@pytest.mark.asyncio
async def test_non_tabular_invocations_do_not_expose_records_artifacts(tmp_path):
    artifact_name = "records_0123456789ABCDEFGHIJK.jsonl"
    (tmp_path / artifact_name).write_text("{}\n", encoding="utf-8")
    app = InferInvocationAPP(
        lambda: {},
        EmptyInputDataModel,
        BaseModel,
        records_output_dir=tmp_path,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app.app), base_url="http://test"
    ) as client:
        response = await client.get(f"/invocations/artifacts/{artifact_name}")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_invocations_rejects_non_records_artifact_names(tmp_path):
    artifact_name = "unrelated.jsonl"
    (tmp_path / artifact_name).write_text("{}\n", encoding="utf-8")
    app = InferInvocationAPP(
        lambda: {
            "tables": [{"id": "table_1", "file_name": "infer.csv"}],
            "categories": [],
            "regressions": [],
            "records": {
                "format": "jsonl",
                "href": f"/invocations/artifacts/{artifact_name}",
                "rows": 1,
                "bytes": 3,
            },
            "meta": {"window_size": 1},
        },
        EmptyInputDataModel,
        AiCOCOTabularOutputDataModel,
        records_output_dir=tmp_path,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app.app), base_url="http://test"
    ) as client:
        response = await client.get(f"/invocations/artifacts/{artifact_name}")

    assert response.status_code == 404


def test_tabular_inference_model_supports_legacy_output_formatter_signature(tmp_path):
    model = LegacyTabularInferenceModel()
    model.formatter = AiCOCOTabularClassificationOutputStrategy()

    result = model(
        tables=[{"id": "table_1", "file_name": "infer.csv"}],
        meta={"window_size": 1},
        files=[str(tmp_path / "infer.csv")],
        records_output_dir=tmp_path,
    )

    assert result.records.rows == 1
