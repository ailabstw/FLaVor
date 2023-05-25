#!/usr/bin/env python
import argparse
import os

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LOGLEVEL"] = "ERROR"
os.environ["FLAVOR"] = "true"
os.environ["CHECK_FV"] = "true"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--preprocess", type=str, help="data preprocess command", default=None
    )
    parser.add_argument("-m", "--main", type=str, required=True, help="main process command")
    parser.add_argument("-y", "--yes", action="store_true", help="skip setting env", default=False)
    args, unparsed = parser.parse_known_args()

    input_path_default = os.getenv("INPUT_PATH", "/data")
    output_path_default = os.getenv("OUTPUT_PATH", "/output")
    weight_path_default = os.getenv("WEIGHT_PATH", "/weight/weight.ckpt")
    log_path_default = os.getenv("LOG_PATH", "/log")

    if not args.yes:
        input_path = input(
            "Set $INPUT_PATH: (Press ENTER if using default env - {})".format(input_path_default)
        )
        output_path = input(
            "Set $OUTPUT_PATH: (Press ENTER if using default env - {})".format(output_path_default)
        )
        weight_path = input(
            "Set $WEIGHT_PATH: (Press ENTER if using default env - {})".format(weight_path_default)
        )
        log_path = input(
            "Set $LOG_PATH: (Press ENTER if using default env - {})".format(log_path_default)
        )

        os.environ["INPUT_PATH"] = input_path if input_path else input_path_default
        os.environ["OUTPUT_PATH"] = output_path if output_path else output_path_default
        os.environ["WEIGHT_PATH"] = weight_path if weight_path else weight_path_default
        os.environ["LOG_PATH"] = log_path if log_path else log_path_default
    else:
        os.environ["INPUT_PATH"] = input_path_default
        os.environ["OUTPUT_PATH"] = output_path_default
        os.environ["WEIGHT_PATH"] = weight_path_default
        os.environ["LOG_PATH"] = log_path_default

    os.environ["SCHEMA_PATH"] = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../schema/FVresult.json"
    )

    os.makedirs(os.environ["LOG_PATH"], exist_ok=True)
    os.makedirs(os.environ["OUTPUT_PATH"], exist_ok=True)

    from flavor.taste.servicer import EdgeEvalServicer

    eval_service = EdgeEvalServicer(mainProcess=args.main, preProcess=args.preprocess)
    eval_service.start()

    print("Run Successfullly !!!")


if __name__ == "__main__":

    main()
