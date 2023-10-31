import json
from json import JSONDecodeError
from typing import Any, Dict, List

from pydantic import TypeAdapter
from starlette.datastructures import FormData

from ..models import AiImage
from .base_strategy import BaseStrategy


class AiCOCOInputStrategy(BaseStrategy):
    async def apply(self, form_data: FormData):

        ta = TypeAdapter(List[AiImage])
        try:
            images = json.loads(form_data.get("images").replace("'", '"'))
        except TypeError as e:
            raise JSONDecodeError(doc="", msg=str(e), pos=-1)

        ta.validate_python(images)

        files = form_data.get("files")

        for image, file in zip(images, files):
            image["physical_file_name"] = file

        return {"images": images}


class AiCOCOOutputStrategy(BaseStrategy):
    async def apply(self, result: Dict[str, Any]):

        # ta = TypeAdapter(AiCOCOFormat)

        # ta.validate_python(result)

        return result
