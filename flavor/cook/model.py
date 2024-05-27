import numbers
from typing import List

from pydantic import BaseModel, Extra, Field, model_validator


class LocalTrainRequest(BaseModel):
    EpR: int


class Metrics(BaseModel):
    confusion_fn: int = Field(..., alias="basic/confusion_fn")
    confusion_fp: int = Field(..., alias="basic/confusion_fp")
    confusion_tn: int = Field(..., alias="basic/confusion_tn")
    confusion_tp: int = Field(..., alias="basic/confusion_tp")

    class Config:
        extra = Extra.allow

    @model_validator(mode="after")
    def check_extra_fields(self):
        for key, value in self.model_extra.items():
            if not isinstance(value, numbers.Number):
                raise ValueError(f"Extra field {key} must be a number")


class Metadata(BaseModel):
    class Config:
        extra = Extra.forbid

    datasetSize: int
    epoch: int
    importance: float


class LocalModel(BaseModel):
    class Config:
        extra = Extra.forbid

    path: str
    metadata: Metadata
    metrics: Metrics


class AggregatedModel(BaseModel):
    class Config:
        extra = Extra.forbid

    path: str


class AggregateRequest(BaseModel):
    class Config:
        extra = Extra.forbid

    AggregatedModel: AggregatedModel
    LocalModels: List[LocalModel]


class FLResponse(BaseModel):
    class Config:
        extra = Extra.forbid

    metadata: Metadata
    metrics: Metrics
