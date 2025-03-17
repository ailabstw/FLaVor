import glob
import json
import os
from pathlib import Path

import pytest
import requests
from httpx import ASGITransport, AsyncClient


def download_file_from_google_drive(file_id: str, destination: str) -> None:
    """
    Download a file from Google Drive using the given file_id and save it to destination.
    """
    url = "https://docs.google.com/uc?export=download&confirm=1"
    session = requests.Session()
    response = session.get(url, params={"id": file_id}, stream=True)
    token = _get_confirm_token(response)
    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(url, params=params, stream=True)
    _save_response_content(response, destination)


def _get_confirm_token(response) -> str:
    """
    Retrieve the confirmation token from response cookies.
    """
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return ""


def _save_response_content(response, destination: str) -> None:
    """
    Save the response content in chunks to the destination file.
    """
    chunk_size = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def load_and_stringify_json(file_path: str) -> dict:
    """
    Load JSON from file_path and return a dict in which each value is a JSON-stringified string.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return {k: json.dumps(v) for k, v in data.items()}


def prepare_test_files(src: str, file_id: str) -> None:
    """
    Check if test files exist. If not, download and extract them.
    """
    input_files = glob.glob(src + "/*")
    if not input_files:
        print("Downloading multiple dicom files...")
        download_file_from_google_drive(file_id, src + ".zip")
        os.makedirs(src, exist_ok=True)
        import zipfile
        with zipfile.ZipFile(src + ".zip", "r") as zip_ref:
            zip_ref.extractall(src)
        os.remove(src + ".zip")


def prepare_nifti_file(file_path: str, file_id: str) -> None:
    """
    Check if the Nifti file exists. If not, download it.
    """
    if not os.path.exists(file_path):
        print("Downloading Nifti file...")
        download_file_from_google_drive(file_id, file_path)


@pytest.mark.asyncio
async def test_seg3d():
    seg3d_example = pytest.importorskip("test_tasks.seg3d_example")
    seg3d_app = seg3d_example.app

    # Prepare multiple DICOM files
    dicom_src = "examples/inference/test_data/seg/img0062"
    dicom_file_id = "1h23vhCuUIKJkFw6jC7VV2XU9lGFuxrLw"
    prepare_test_files(dicom_src, dicom_file_id)

    # Prepare single nifti file
    nifti_file = "examples/inference/test_data/seg/img0062.nii.gz"
    nifti_file_id = "1dgvHBlNtuzRON2NsTvykSQjtOG7KBuyX"
    prepare_nifti_file(nifti_file, nifti_file_id)

    # Create data payloads
    data = load_and_stringify_json("examples/inference/test_data/seg/input_3d_dcm.json")
    data_shuffle = load_and_stringify_json("examples/inference/test_data/seg/input_3d_dcm_shuffle.json")
    data_volume = load_and_stringify_json("examples/inference/test_data/seg/input_3d.json")

    # Manipulate shuffle data to drop and cause a mismatch
    data_shuffle_drop = data_shuffle.copy()
    for k in data_shuffle_drop:
        data_list = json.loads(data_shuffle_drop[k])
        data_shuffle_drop[k] = json.dumps(data_list[10:])  # drop first 10 items

    # Collect file paths
    dicom_files = glob.glob(dicom_src + "/*")

    # Open DICOM files
    dicom_file_tuples = []
    for fpath in dicom_files:
        f = open(fpath, "rb")
        dicom_file_tuples.append(("files", (Path(fpath).name, f)))

    # Open Nifti file
    nifti_f = open(nifti_file, "rb")
    nifti_file_tuples = [("files", (Path(nifti_file).name, nifti_f))]

    async with AsyncClient(
        transport=ASGITransport(app=seg3d_app.app), base_url="http://test"
    ) as client:
        response = await client.post("/invocations", data=data, files=dicom_file_tuples)
        response_shuffle = await client.post("/invocations", data=data_shuffle, files=dicom_file_tuples)
        response_shuffle_drop = await client.post("/invocations", data=data_shuffle_drop, files=dicom_file_tuples)
        response_volume = await client.post("/invocations", data=data_volume, files=nifti_file_tuples)

        assert response.status_code == 200
        assert response_shuffle.status_code == 200
        # The mismatch test
        assert response_shuffle_drop.status_code == 400
        assert response_volume.status_code == 200

        ordered_content = json.loads(response.content)
        unordered_content = json.loads(response_shuffle.content)
        volumetric_content = json.loads(response_volume.content)

        for ordered, unordered, volumetric in zip(
            ordered_content["annotations"],
            unordered_content["annotations"],
            volumetric_content["annotations"],
        ):
            assert (
                ordered["segmentation"] == unordered["segmentation"] == volumetric["segmentation"]
            )

        for ordered, unordered, volumetric in zip(
            ordered_content["images"],
            unordered_content["images"],
            volumetric_content["images"],
        ):
            assert len(ordered) == len(unordered) == len(volumetric)

    # Close all file handles
    for _, (_, fobj) in dicom_file_tuples:
        fobj.close()
    nifti_f.close()
    