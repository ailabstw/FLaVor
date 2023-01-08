import json
import os
import time

EventSet = {"TrainInitDone", "TrainStarted", "TrainFinished", "Error"}
event_path = "/output"


def SetEvent(event: str):
    if event not in EventSet:
        raise ValueError("Unknown event {}".format(event))
    os.makedirs(event_path, exist_ok=True)
    open(os.path.join(event_path, event), "w").close()


def WaitEvent(event: str):
    if event not in EventSet:
        raise ValueError("Unknown event {}".format(event))
    while not os.path.exists(os.path.join(event_path, event)):
        time.sleep(1)
    os.remove(os.path.join(event_path, event))


def CleanAllEvent():
    for event in EventSet:
        if os.path.exists(os.path.join(event_path, event)):
            os.remove(os.path.join(event_path, event))


def IsSetEvent(event: str):
    if event not in EventSet:
        raise ValueError("Unknown event {}".format(event))
    return os.path.exists(os.path.join(event_path, event))


def SaveInfoJson(info: dict):
    with open(
        os.path.join(os.path.dirname(os.environ["LOCAL_MODEL_PATH"]), "info.json"), "w"
    ) as openfile:
        json.dump(info, openfile)


def CleanInfoJson():
    info_path = os.path.join(os.path.dirname(os.environ["LOCAL_MODEL_PATH"]), "info.json")
    if os.path.exists(info_path):
        os.remove(os.path.join(info_path))
