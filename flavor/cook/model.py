from typing import List

from pydantic import BaseModel, Extra, Field


class LocalTrainRequest(BaseModel):
    EpR: int


class Metrics(BaseModel):
    confusion_fn: int = Field(..., alias="basic/confusion_fn")
    confusion_fp: int = Field(..., alias="basic/confusion_fp")
    confusion_tn: int = Field(..., alias="basic/confusion_tn")
    confusion_tp: int = Field(..., alias="basic/confusion_tp")
    precision: float = 0.0

    class Config:
        allow_population_by_field_name = True
        extra = Extra.allow


class Metadata(BaseModel):
    datasetSize: int
    epoch: int
    importance: float


class LocalModel(BaseModel):
    path: str
    metadata: Metadata
    metrics: Metrics


class AggregatedModel(BaseModel):
    path: str


class AggregateRequest(BaseModel):
    AggregatedModel: AggregatedModel
    LocalModels: List[LocalModel]
