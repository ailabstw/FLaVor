import glob
import json
import os
from pathlib import Path

import pytest
import requests
from httpx import AsyncClient


def download_file_from_google_drive(file_id, destination):
    # please refer to https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


@pytest.mark.asyncio
async def test_seg3d():
    from test_tasks.seg3d_example import app

    src = "examples/inference/test_data/seg/img0062"
    input_files = glob.glob(src + "/*")
    if not input_files:
        print("Testing input files are not found. Downloading...")
        download_file_from_google_drive("1h23vhCuUIKJkFw6jC7VV2XU9lGFuxrLw", src + ".zip")
        print("Download complete")
        os.makedirs(src, exist_ok=True)
        import zipfile

        with zipfile.ZipFile(src + ".zip", "r") as zip_ref:
            zip_ref.extractall(src)
        os.remove(src + ".zip")

    files = []
    for filepath in input_files:
        filepath = Path(filepath)
        file = open(filepath, "rb")
        files.append(("files", (f"_{filepath.parent.stem}_{filepath.name}", file)))

    with open("examples/inference/test_data/seg/input_3d_dcm.json", "r") as f:
        data = json.load(f)

    for k in data:
        data[k] = json.dumps(data[k])

    async with AsyncClient(app=app.app, base_url="http://test") as client:
        response = await client.post("/invocations", data=data, files=files)
        assert response.status_code == 200
