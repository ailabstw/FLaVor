from typing import List, Optional

from pydantic import BaseModel, Extra, Field


class Metadata(BaseModel):
    class Config:
        extra = Extra.forbid

    datasetSize: int


class Bar(BaseModel):
    class Config:
        extra = Extra.forbid

    title: str
    labels: List[str]
    values: List[float]
    y_axis: Optional[str] = Field(None, alias="y-axis")


class Heatmap(BaseModel):
    class Config:
        extra = Extra.forbid

    title: str
    x_labels: List[str] = Field(..., alias="x-labels")
    y_labels: List[str] = Field(..., alias="y-labels")
    values: List[List[float]]
    x_axis: str = Field(..., alias="x-axis")
    y_axis: str = Field(..., alias="y-axis")


class Image(BaseModel):
    class Config:
        extra = Extra.forbid

    title: str
    filename: str


class Plot(BaseModel):
    class Config:
        extra = Extra.forbid

    title: str
    labels: List[str]
    x_values: List[List[float]] = Field(..., alias="x-values")
    y_values: List[List[float]] = Field(..., alias="y-values")
    x_axis: str = Field(..., alias="x-axis")
    y_axis: str = Field(..., alias="y-axis")


class Results(BaseModel):
    class Config:
        extra = Extra.forbid

    tables: Optional[List[Bar]] = None
    bars: Optional[List[Bar]] = None
    heatmaps: Optional[List[Heatmap]] = None
    plots: Optional[List[Plot]] = None
    images: Optional[List[Image]] = None


class FVResponse(BaseModel):
    class Config:
        extra = Extra.forbid

    metadata: Metadata
    results: Results
