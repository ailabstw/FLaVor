from .aicoco_strategy import (
    AiCOCOClassificationOutputStrategy,
    AiCOCODetectionOutputStrategy,
    AiCOCORegressionOutputStrategy,
    AiCOCOSegmentationOutputStrategy,
)
from .gradio_strategy import GradioDetectionStrategy, GradioSegmentationStrategy

__all__ = [
    "AiCOCOClassificationOutputStrategy",
    "AiCOCODetectionOutputStrategy",
    "AiCOCORegressionOutputStrategy",
    "AiCOCOSegmentationOutputStrategy",
    "GradioDetectionStrategy",
    "GradioSegmentationStrategy",
]
