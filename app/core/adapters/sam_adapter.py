import logging
from typing import Any, Dict

import numpy as np

from app.core.adapters.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class SamAdapter(BaseAdapter):
    """Adapter for SAM (Segment Anything Model) via ultralytics.

    Wraps extractor.SamSegmenter using the same load/predict/unload
    conventions as YoloAdapter and OcrAdapter.

    predict() input: {"frame": np.ndarray (BGR), "box": (x, y, w, h)}
    predict() output: np.ndarray mask (uint8, 0 or 255, full-frame size)
    """

    def load_model(self, model_path: str, parameters: Dict[str, Any]) -> bool:
        try:
            logger.info(f"SamAdapter loading model from {model_path}")
            from app.modules.target_extraction.extractor import SamSegmenter
            self.model = SamSegmenter(weights=model_path)
            # Trigger lazy load now so failures surface here
            self.model._load()
            self.is_loaded = True
            return True
        except Exception as e:
            logger.error(f"SamAdapter failed to load model: {e}")
            self.is_loaded = False
            return False

    def predict(self, input_data: Any) -> Any:
        if not self.is_loaded or self.model is None:
            raise RuntimeError("SAM model is not loaded.")
        frame: np.ndarray = input_data["frame"]
        box = input_data["box"]  # (x, y, w, h)
        return self.model.segment(frame, box)
