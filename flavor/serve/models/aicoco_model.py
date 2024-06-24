from enum import Enum
from typing import Optional, Sequence

from pydantic import BaseModel


class Iscrowd(Enum):
    number_0 = 0
    number_1 = 1


class AiAnnotation(BaseModel):
    bbox: Optional[Sequence[Sequence[int]]] = None
    id: str
    image_id: str
    iscrowd: Iscrowd
    object_id: str
    segmentation: Optional[Sequence[Sequence[int]]] = None


class AiCategory(BaseModel):
    id: str
    name: str
    supercategory_id: Optional[str] = None
    color: Optional[str] = None


class AiRegression(BaseModel):
    id: str
    name: str
    superregression_id: Optional[str] = None
    unit: Optional[str] = None
    threshold: Optional[str] = None


class AiRegressionItem(BaseModel, extra="forbid"):
    regression_id: str
    value: float


class AiObject(BaseModel):
    category_ids: Optional[Sequence[str]] = None
    confidence: Optional[float] = None
    id: str
    regressions: Optional[Sequence[AiRegressionItem]] = None


class TaskType(Enum):
    binary = "binary"
    multilabel = "multilabel"
    multiclass = "multiclass"


class AiMeta(BaseModel):
    category_ids: Optional[Sequence[str]] = None
    regressions: Optional[Sequence[AiRegressionItem]] = None
    task_type: Optional[TaskType] = None


class AiImage(BaseModel, extra="allow"):
    category_ids: Optional[Sequence[str]] = None
    file_name: str
    id: str
    index: int
    regressions: Optional[Sequence[AiRegressionItem]] = None


class AiCOCOFormat(BaseModel, extra="forbid"):
    images: Sequence[AiImage]
    annotations: Sequence[AiAnnotation]
    categories: Sequence[AiCategory]
    regressions: Sequence[AiRegression]
    objects: Sequence[AiObject]
    meta: AiMeta
