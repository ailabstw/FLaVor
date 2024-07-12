from enum import Enum
from typing import Optional, Sequence, Union

from pydantic import BaseModel


class Iscrowd(Enum):
    number_0 = 0
    number_1 = 1


class AiAnnotation(BaseModel):
    id: str
    image_id: str
    iscrowd: Iscrowd
    object_id: str
    bbox: Union[Sequence[Sequence[int]], None]
    segmentation: Union[Sequence[Sequence[int]], None]


class AiCategory(BaseModel):
    id: str
    name: str
    supercategory_id: Union[str, None]
    color: Optional[str] = None


class AiRegression(BaseModel):
    id: str
    name: str
    superregression_id: Union[str, None]
    unit: Optional[str] = None
    threshold: Optional[str] = None


class AiRegressionItem(BaseModel, extra="forbid"):
    regression_id: str
    value: float


class AiObject(BaseModel):
    id: str
    category_ids: Union[Sequence[str], None]
    regressions: Union[Sequence[AiRegressionItem], None]
    confidence: Optional[float] = None


class TaskType(Enum):
    binary = "binary"
    multilabel = "multilabel"
    multiclass = "multiclass"


class AiMeta(BaseModel):
    category_ids: Union[Sequence[str], None]
    regressions: Union[Sequence[AiRegressionItem], None]
    task_type: Optional[TaskType] = None


class AiImage(BaseModel, extra="allow"):
    file_name: str
    id: str
    index: int
    category_ids: Union[Sequence[str], None]
    regressions: Union[Sequence[AiRegressionItem], None]


class AiCOCOFormat(BaseModel, extra="forbid"):
    images: Sequence[AiImage]
    annotations: Sequence[AiAnnotation]
    categories: Sequence[AiCategory]
    regressions: Sequence[AiRegression]
    objects: Sequence[AiObject]
    meta: AiMeta


# TODO add Image predix
