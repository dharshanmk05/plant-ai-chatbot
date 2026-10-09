from PIL import Image
import numpy as np

from services.disease_predictor import PlantDiseasePredictor
from services.image_utils import basic_image_quality_check, prepare_image_for_model


def test_quality_check_accepts_good_image():
    image = Image.new("RGB", (300, 300), color=(120, 180, 90))
    result = basic_image_quality_check(image)
    assert result["is_valid"] is True


def test_prepare_image_for_model_returns_expected_shape():
    image = Image.new("RGB", (224, 224), color=(50, 150, 70))
    processed = prepare_image_for_model(image)
    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32
    assert processed.max() == 150


def test_prediction_result_structure():
    result = PlantDiseasePredictor.build_prediction_result(
        "Tomato Early Blight",
        0.91,
        [{"label": "Tomato Early Blight", "score": 0.91}],
        threshold=0.60,
    )
    assert result["plant"] == "Tomato"
    assert result["disease"] == "Early Blight"
    assert result["healthy"] is False
    assert 0.90 <= result["confidence"] <= 1.0


def test_plant_disease_model_labels_are_parsed():
    result = PlantDiseasePredictor.build_prediction_result(
        "Corn (Maize) with Common Rust",
        0.91,
        [{"label": "Corn (Maize) with Common Rust", "score": 0.91}],
    )
    assert result["plant"] == "Maize"
    assert result["disease"] == "Common Rust"


def test_healthy_disease_model_label_is_parsed():
    result = PlantDiseasePredictor.build_prediction_result(
        "Healthy Tomato Plant",
        0.91,
        [{"label": "Healthy Tomato Plant", "score": 0.91}],
    )
    assert result["plant"] == "Tomato"
    assert result["disease"] == "Healthy"
    assert result["healthy"] is True
