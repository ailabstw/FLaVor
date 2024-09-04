import json

import torch

from flavor.cook.utils import GetPaths, SetEvent, WaitEvent


class FedAvg(object):
    def aggregate(self, factors, localModelPaths):

        globalModel = {}

        for idx, localModelPath in enumerate(localModelPaths):

            localModel = torch.load(localModelPath)["state_dict"]

            for weight_key in localModel:
                globalModel[weight_key] = (
                    localModel[weight_key] * factors[idx]
                    if idx == 0
                    else globalModel[weight_key] + localModel[weight_key] * factors[idx]
                )

        return globalModel

    def __call__(self):

        # Get Path
        localModelPaths = GetPaths("localModels")
        localInfoPaths = GetPaths("localInfos")
        globalModelPath = GetPaths("globalModel")[0]
        globalInfoPath = GetPaths("globalInfo")[0]  # noqa 841

        # Caluculate aggregation factors
        datasetSize = []
        for localInfoPath in localInfoPaths:
            with open(localInfoPath, "r") as openfile:
                localInfo = json.load(openfile)
            datasetSize.append(float(localInfo["metadata"]["datasetSize"]))

        factors = [d / sum(datasetSize) for d in datasetSize]

        # Aggregate weights
        globalModel = self.aggregate(factors, localModelPaths)

        # Save model
        torch.save({"state_dict": globalModel}, globalModelPath)


def main_init_once():

    aggregator = FedAvg()

    while True:
        WaitEvent("AggregateStarted")
        aggregator()
        SetEvent("AggregateFinished")

    # Notify the external service that the process is finished.
    SetEvent("ProcessFinished")


def main_init_every_round():

    aggregator = FedAvg()
    aggregator()


if __name__ == "__main__":

    # Depend on the way user implementation
    main_init_every_round()  # or main_init_once()
