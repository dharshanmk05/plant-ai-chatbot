from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from model.model_config import CACHE_DIR, CONFIDENCE_THRESHOLD, MAX_IMAGE_DIMENSION, MODEL_NAME
from services.image_utils import basic_image_quality_check, ensure_rgb, prepare_image_for_model


class PlantDiseasePredictor:
    """Reusable predictor class for public Hugging Face image classifiers."""

    def __init__(self, model_name: str = MODEL_NAME, cache_dir: str = CACHE_DIR, threshold: float = CONFIDENCE_THRESHOLD) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threshold = threshold
        self.model = None
        self.processor = None

    @staticmethod
    def _label_to_text(label: Any) -> str:
        text = str(label or "").strip()
        if not text:
            return "Unclear plant condition"
        return text.replace("_", " ").replace("-", " ").title()

    @staticmethod
    def _infer_plant_name(label: str) -> str:
        label_lower = label.lower()
        if "apple" in label_lower:
            return "Apple"
        if "blueberry" in label_lower:
            return "Blueberry"
        if "cherry" in label_lower:
            return "Cherry"
        if "tomato" in label_lower:
            return "Tomato"
        if "potato" in label_lower:
            return "Potato"
        if "pepper" in label_lower or "chilli" in label_lower:
            return "Pepper"
        if "orange" in label_lower:
            return "Orange"
        if "peach" in label_lower:
            return "Peach"
        if "raspberry" in label_lower:
            return "Raspberry"
        if "squash" in label_lower:
            return "Squash"
        if "strawberry" in label_lower:
            return "Strawberry"
        if "paddy" in label_lower or "rice" in label_lower:
            return "Paddy"
        if "brinjal" in label_lower or "eggplant" in label_lower:
            return "Brinjal"
        if "coriander" in label_lower:
            return "Coriander"
        if "onion" in label_lower:
            return "Onion"
        if "okra" in label_lower or "lady finger" in label_lower:
            return "Okra"
        if "wheat" in label_lower:
            return "Wheat"
        if "maize" in label_lower or "corn" in label_lower:
            return "Maize"
        if "banana" in label_lower:
            return "Banana"
        if "groundnut" in label_lower or "peanut" in label_lower:
            return "Groundnut"
        if "sugarcane" in label_lower or "sugar cane" in label_lower:
            return "Sugarcane"
        if "cotton" in label_lower:
            return "Cotton"
        if "cucumber" in label_lower:
            return "Cucumber"
        if "cabbage" in label_lower:
            return "Cabbage"
        if "cauliflower" in label_lower:
            return "Cauliflower"
        if "mango" in label_lower:
            return "Mango"
        if "grape" in label_lower or "grapevine" in label_lower:
            return "Grapevine"
        if "soybean" in label_lower or "soy bean" in label_lower:
            return "Soybean"
        if "sunflower" in label_lower:
            return "Sunflower"
        if "coconut" in label_lower:
            return "Coconut"
        return "Plant"

    @staticmethod
    def _infer_disease_name(label: str, healthy: bool) -> str:
        if healthy:
            return "Healthy"

        normalized = label.lower().replace("_", " ").replace("-", " ")
        disease_labels = [
            ("cedar apple rust", "Cedar Apple Rust"),
            ("cercospora and gray leaf spot", "Gray Leaf Spot"),
            ("northern leaf blight", "Northern Leaf Blight"),
            ("yellow leaf curl virus", "Tomato Yellow Leaf Curl Virus"),
            ("two spotted spider mite", "Spider Mites"),
            ("spider mites", "Spider Mites"),
            ("citrus greening", "Citrus Greening"),
            ("bacterial spot", "Bacterial Spot"),
            ("early blight", "Early Blight"),
            ("late blight", "Late Blight"),
            ("septoria leaf spot", "Septoria Leaf Spot"),
            ("powdery mildew", "Powdery Mildew"),
            ("common rust", "Common Rust"),
            ("black rot", "Black Rot"),
            ("leaf mold", "Leaf Mold"),
            ("leaf scorch", "Leaf Scorch"),
            ("target spot", "Target Spot"),
            ("isariopsis leaf spot", "Isariopsis Leaf Spot"),
            ("black measles", "Esca"),
            ("esca", "Esca"),
            ("mosaic virus", "Mosaic Virus"),
            ("apple scab", "Apple Scab"),
        ]
        for match, disease in disease_labels:
            if match in normalized:
                return disease

        return PlantDiseasePredictor._label_to_text(label)

    @staticmethod
    def build_prediction_result(label: str, confidence: float, top_predictions: list[dict[str, Any]], threshold: float = CONFIDENCE_THRESHOLD) -> dict[str, Any]:
        text = PlantDiseasePredictor._label_to_text(label)
        lower_label = text.lower()

        healthy = "healthy" in lower_label or "good" in lower_label or "normal" in lower_label
        disease = PlantDiseasePredictor._infer_disease_name(text, healthy)
        plant = PlantDiseasePredictor._infer_plant_name(text)

        score = float(max(0.0, min(1.0, confidence)))
        return {
            "plant": plant,
            "disease": disease,
            "confidence": round(score, 4),
            "healthy": healthy,
            "top_predictions": top_predictions,
            "threshold": threshold,
        }

    def load_model(self) -> None:
        """Load the image-classification model and processor lazily only when needed."""
        if self.model is not None and self.processor is not None:
            return

        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model.eval()
        except Exception as exc:  # pragma: no cover - network/loading path is handled by the app
            raise RuntimeError(
                "The Hugging Face model could not be loaded. Check your internet connection and try again."
            ) from exc

    def analyze(self, image: Any) -> dict[str, Any]:
        """Validate image quality, run inference, and return the top prediction."""
        if image is None:
            raise ValueError("No image was uploaded.")

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype("uint8"))

        rgb_image = ensure_rgb(image)
        quality = basic_image_quality_check(rgb_image)
        if not quality["is_valid"]:
            raise ValueError(quality["message"])

        self.load_model()
        processed = prepare_image_for_model(rgb_image, target_size=(MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
        inputs = self.processor(images=processed, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            top_values, top_indexes = torch.topk(probabilities, k=min(5, probabilities.shape[-1]))

        id_to_label = getattr(self.model.config, "id2label", {})
        top_predictions = []
        for index, score in zip(top_indexes[0].tolist(), top_values[0].tolist()):
            label = id_to_label.get(index, f"Class {index}")
            top_predictions.append({"label": label, "score": float(score)})

        best_label = top_predictions[0]["label"]
        best_score = top_predictions[0]["score"]
        prediction = self.build_prediction_result(best_label, best_score, top_predictions, threshold=self.threshold)

        if prediction["confidence"] < self.threshold:
            prediction["status"] = "Unable to make a reliable prediction"
        elif prediction["healthy"]:
            prediction["status"] = "Healthy"
        else:
            prediction["status"] = "Disease Detected"

        return prediction
