import argparse
import glob
import json
import os
import time
from pathlib import Path

import requests

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--filenames", type=str, action="append", required=True, help="Input file dir"
)
parser.add_argument("-d", "--data", type=str, required=True, help="Json file path")
parser.add_argument("-p", "--port", type=str, default=9111, help="Port")
args = parser.parse_args()

while True:
    try:
        res = requests.get("http://0.0.0.0:{}/ping".format(os.getenv("PORT", args.port)))
        if res.status_code == 204:
            break
    except Exception:
        time.sleep(1)
        pass

filepaths = []
for filename in args.filenames:
    filepaths.extend(glob.glob(filename))

files = []

for filepath in filepaths:
    filepath = Path(filepath)
    file = open(filepath, "rb")
    files.append(("files", (f"{filepath.name}", file)))

with open(args.data, "r") as f:
    data = json.load(f)

for k in data:
    data[k] = json.dumps(data[k])

recv = requests.post(
    "http://0.0.0.0:{}/invocations".format(os.getenv("PORT", args.port)),
    data=data,
    files=files,
)

print(recv.json())

for field, file in files:
    file[1].close()
