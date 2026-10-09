from services.chatbot import PlantCareChatbot


def test_disease_answer_describes_the_affected_signs():
    prediction = {
        "plant": "Tomato",
        "disease": "Early Blight",
        "confidence": 0.56,
        "threshold": 0.60,
        "healthy": False,
    }

    answer = PlantCareChatbot().generate_response("What disease is this?", prediction)

    assert "possible match: Tomato Early Blight" in answer
    assert "fungal" in answer
    assert "lower leaves" in answer.lower()
    assert "confirm the match before treatment" in answer


def test_medicine_answer_avoids_unverified_products_and_requests_region():
    prediction = {
        "plant": "Tomato",
        "disease": "Early Blight",
        "confidence": 0.90,
        "threshold": 0.60,
        "healthy": False,
    }

    answer = PlantCareChatbot().generate_response("What medicine should I use?", prediction)

    assert "Treatment guidance for Tomato Early Blight" in answer
    assert "does not verify a specific medicine name or dose" in answer
    assert "country or region" in answer