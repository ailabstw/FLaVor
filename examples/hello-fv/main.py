from flavors.taste.servicer import EdgeEvalServicer


def main():

    eval_service = EdgeEvalServicer()
    eval_service.dataSubProcess = None
    eval_service.valSubProcess = "python test.py"

    eval_service.start()


if __name__ == "__main__":

    main()
