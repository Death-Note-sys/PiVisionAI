import logging
import time
from typing import Dict, Any
from app.core.adapters.base_adapter import BaseAdapter
import easyocr

logger = logging.getLogger(__name__)

class OcrAdapter(BaseAdapter):
    """Adapter for the EasyOCR framework. Unlike YOLO, EasyOCR has no
    weights file path to load — it manages its own model downloads
    internally, keyed by a language list. model_path is accepted for
    interface compatibility with IAdapter but is not used."""

    def load_model(self, model_path: str, parameters: Dict[str, Any]) -> bool:
        try:
            languages = parameters.get("languages", ["en"])
            gpu = parameters.get("gpu", False)
            logger.info(f"OcrAdapter loading EasyOCR reader (languages={languages}, gpu={gpu})")
            self.model = easyocr.Reader(languages, gpu=gpu)
            self.is_loaded = True
            return True
        except Exception as e:
            logger.error(f"OcrAdapter failed to load: {e}")
            self.is_loaded = False
            return False

    def predict(self, input_data: Any) -> Any:
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded.")

        frame = input_data.get("frame")
        min_confidence = input_data.get("min_confidence", 0.3)

        start_time = time.perf_counter()
        raw_results = self.model.readtext(frame)
        end_time = time.perf_counter()

        texts = []
        for (bbox, text, confidence) in raw_results:
            if confidence < min_confidence:
                continue
            points = [[int(pt[0]), int(pt[1])] for pt in bbox]
            texts.append({
                "text": text,
                "confidence": float(confidence),
                "points": points,
            })

        return {
            "texts": texts,
            "latency_ms": (end_time - start_time) * 1000,
        }
