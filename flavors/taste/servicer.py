import json
import logging
import subprocess
import sys


class EdgeEvalServicer(object):
    def __init__(self):

        self.__log_filename = "/log/error.log"
        self.__progress_file = "/log/progress.json"
        with open(self.__progress_file, "w") as f:
            json.dump({"status": "", "completedPercentage": ""}, f, indent=2)

        self.dataSubProcess = None
        self.valSubProcess = None

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
        sys.exit()


    def start(self):

        # 1. initialization
        self.update_progress("initialization", 25)

        # 2. preprocessing
        self.update_progress("preprocessing", 50)
        if self.dataSubProcess:
            try:
                logging.info("Start data preprocessing.")
                subprocess.check_output(
                    [ele for ele in self.dataSubProcess.split(" ") if ele.strip()],
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
                [ele for ele in self.valSubProcess.split(" ") if ele.strip()],
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            self.terminate(e.output.decode("utf-8"))
        logging.info("Complete validation.")

        # 4. completed
        self.update_progress("completed", 100)
