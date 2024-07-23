from .aicoco_strategy import (
    AiCOCOClassificationOutputStrategy,
    AiCOCODetectionOutputStrategy,
    AiCOCORegressionOutputStrategy,
    AiCOCOSegmentationOutputStrategy,
    AiCOCOTabularClassificationOutputStrategy,
    AiCOCOTabularRegressionOutputStrategy,
)
from .gradio_strategy import GradioDetectionStrategy, GradioSegmentationStrategy

__all__ = [
    "AiCOCOClassificationOutputStrategy",
    "AiCOCODetectionOutputStrategy",
    "AiCOCORegressionOutputStrategy",
    "AiCOCOSegmentationOutputStrategy",
    "AiCOCOTabularClassificationOutputStrategy",
    "AiCOCOTabularRegressionOutputStrategy" "GradioDetectionStrategy",
    "GradioSegmentationStrategy",
]
