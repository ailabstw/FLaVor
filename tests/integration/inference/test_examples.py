import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any, List, Optional, Union

import pytest
from httpx import ASGITransport, AsyncClient

from flavor.serve.inference.strategies.aicoco_strategy import set_global_seed


async def post_data_and_files(app, data_path: Path, file_paths: Union[Path, List[Path]]):
    """
    1. Read data_path (a JSON file) as the request body (both keys and values are serialized).
    2. Read one or more file_paths (files) and upload them using multipart/form-data.
    3. Return the entire response object for later assertion.
    """
    data = json.loads(data_path.read_text())
    post_data = {k: json.dumps(v) for k, v in data.items()}

    if isinstance(file_paths, Path):
        file_paths = [file_paths]

    with ExitStack() as stack:
        files = []
        for fp in file_paths:
            f = stack.enter_context(fp.open("rb"))
            files.append(("files", (fp.name, f)))

        async with AsyncClient(
            transport=ASGITransport(app=app.app), base_url="http://test"
        ) as client:
            response = await client.post("/invocations", data=post_data, files=files)
            return response  # Return httpx.Response


def compare_dicts(
    d1: Any, d2: Any, tolerance: float = 1e-2, ignored_keys: Optional[List[str]] = None
) -> bool:
    if ignored_keys is None:
        ignored_keys = []

    if type(d1) != type(d2):
        return False

    if isinstance(d1, dict):
        for key in d1:
            if key in ignored_keys:
                continue
            if key not in d2 or not compare_dicts(d1[key], d2[key], tolerance, ignored_keys):
                return False
        for key in d2:
            if key not in d1 and key not in ignored_keys:
                return False
        return True

    elif isinstance(d1, list):
        if len(d1) != len(d2):
            return False
        return all(compare_dicts(a, b, tolerance, ignored_keys) for a, b in zip(d1, d2))

    elif isinstance(d1, (int, float)):
        return math.isclose(d1, d2, abs_tol=tolerance)

    else:
        return d1 == d2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "import_name, input_json, file_paths, expected_json",
    [
        (
            "test_tasks.seg_example",
            "examples/inference/test_data/seg/input.json",
            "examples/inference/test_data/seg/0.dcm",
            "examples/inference/test_data/seg/expected.json",
        ),
        (
            "test_tasks.cls_example",
            "examples/inference/test_data/cls_reg/input.json",
            "examples/inference/test_data/cls_reg/n02123159_tiger_cat.jpeg",
            "examples/inference/test_data/cls_reg/expected_cls.json",
        ),
        (
            "test_tasks.reg_example",
            "examples/inference/test_data/cls_reg/input.json",
            "examples/inference/test_data/cls_reg/n02123159_tiger_cat.jpeg",
            "examples/inference/test_data/cls_reg/expected_reg.json",
        ),
        (
            "test_tasks.det_example",
            "examples/inference/test_data/det/input.json",
            "examples/inference/test_data/det/5fb00ac1228969a39cee7cd6678ee704.jpg",
            "examples/inference/test_data/det/expected.json",
        ),
        (
            "test_tasks.tabular_cls_example",
            "examples/inference/test_data/tabular/cls/input.json",
            "examples/inference/test_data/tabular/cls/test_cls.csv",
            "examples/inference/test_data/tabular/cls/expected.json",
        ),
        (
            "test_tasks.tabular_reg_example",
            "examples/inference/test_data/tabular/reg/input.json",
            "examples/inference/test_data/tabular/reg/test_reg.csv",
            "examples/inference/test_data/tabular/reg/expected.json",
        ),
        (
            "test_tasks.hybrid_example",
            "examples/inference/test_data/hybrid/input.json",
            [
                "examples/inference/test_data/hybrid/451c164d-7684-44b1-81b2-956247db765b_20160112_102927.jpg",
                "examples/inference/test_data/hybrid/test_cls.csv",
            ],
            "examples/inference/test_data/hybrid/expected.json",
        ),
    ],
)
async def test_inference(import_name, input_json, file_paths, expected_json):
    """
    Test for different inference scenarios (segmentation, classification, regression, detection, hybrid).
    """
    set_global_seed(0)

    # 0) importorskip: Skip the test if the module is not available in the environment
    imported_example = pytest.importorskip(import_name)
    app = imported_example.app

    # 1) Prepare file paths
    if isinstance(file_paths, str):
        file_paths = Path(file_paths)
    elif isinstance(file_paths, list):
        file_paths = [Path(fp) for fp in file_paths]

    # 2) Execute the test request
    response = await post_data_and_files(app, Path(input_json), file_paths)

    # 3) Check status code
    assert response.status_code == 200

    # 4) Load the expected JSON
    expected_data = json.loads(Path(expected_json).read_text())
    actual_data = response.json()

    # 5) Compare response with expected output
    assert compare_dicts(actual_data, expected_data)
