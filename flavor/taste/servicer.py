import json
import logging
import os
import subprocess
import sys

from jsonschema import validate

os.environ["PYTHONWARNINGS"] = "ignore"


class EdgeEvalServicer(object):
    def __init__(self, mainProcess, preProcess):

        self.__log_filename = os.path.join(os.environ["LOG_PATH"], "error.log")
        self.__progress_file = os.path.join(os.environ["LOG_PATH"], "progress.log")
        self.__result_file = os.path.join(os.environ["OUTPUT_PATH"], "result.json")
        with open(self.__progress_file, "w") as f:
            json.dump({"status": "", "completedPercentage": ""}, f, indent=2)

        self.preProcess = preProcess
        self.mainProcess = mainProcess

    def update_progress(self, status, completedPercentage):
        with open(self.__progress_file, "r") as jsonFile:
            data = json.load(jsonFile)

        data["status"] = status
        data["completedPercentage"] = completedPercentage

        with open(self.__progress_file, "w") as jsonFile:
            json.dump(data, jsonFile, indent=2)

    def terminate(self, message):
        logging.error(message)
        with open(self.__log_filename, "w") as F:
            F.write(message)
        raise Exception(message)
        sys.exit()

    def start(self):

        # 1. initialization
        self.update_progress("initialization", 25)
        if os.path.exists(self.__log_filename):
            os.remove(self.__log_filename)

        if os.path.exists(self.__result_file):
            os.remove(self.__result_file)

        # 2. preprocessing
        self.update_progress("preprocessing", 50)
        if self.preProcess:
            try:
                logging.info("Start data preprocessing.")
                subprocess.check_output(
                    [ele for ele in self.preProcess.split(" ") if ele.strip()],
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError as e:
                self.terminate(e.output.decode("utf-8"))
            logging.info("Complete preprocessing.")
        else:
            logging.info("Skip data preprocessing.")

        # 3. validating
        self.update_progress("validating", 75)
        try:
            logging.info("Start validation.")
            subprocess.check_output(
                [ele for ele in self.mainProcess.split(" ") if ele.strip()],
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            self.terminate(e.output.decode("utf-8"))

        if not os.path.exists(self.__result_file):
            self.terminate(
                "The result.json is missing, please check your code! Make sure about the export path and do not catch any errors!"
            )

        try:
            with open(self.__result_file, "r") as openfile:
                instance = json.load(openfile)
            with open(os.environ["SCHEMA_PATH"], "r") as openfile:
                schema = json.load(openfile)
        except Exception as e:
            self.terminate(str(e))

        try:
            validate(instance=instance, schema=schema)
        except Exception:
            self.terminate("Json Schema Error")

        logging.info("Complete validation.")

        # 4. completed
        self.update_progress("completed", 100)
