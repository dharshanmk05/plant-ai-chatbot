from __future__ import annotations

import re
from typing import Any

import requests

from model.model_config import LOCAL_OLLAMA_MODEL, LOCAL_OLLAMA_URL
from services.knowledge_base import get_disease_info, get_plant_info, load_knowledge_base


class PlantCareChatbot:
    """Rule-based plant-care assistant with optional local Ollama support."""

    def __init__(self) -> None:
        self.ollama_url = LOCAL_OLLAMA_URL
        self.model_name = LOCAL_OLLAMA_MODEL

    def is_ollama_available(self) -> bool:
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def generate_response(self, question: str, prediction: dict[str, Any] | None = None, mode: str = "knowledge") -> str:
        """Generate a grounded response based on the current analysis and plant knowledge."""
        if not question or not question.strip():
            return "Please ask a plant care question and I will help based on the current image analysis."

        if mode.lower() == "local_llm" and self.is_ollama_available():
            try:
                return self._generate_ollama_response(question, prediction)
            except Exception:
                pass

        return self._generate_knowledge_based_response(question, prediction)

    def _find_known_plant(self, question: str) -> str | None:
        text = question.strip()
        if not text:
            return None

        knowledge = load_knowledge_base()
        names = [str(entry.get("plant", "")).strip() for entry in knowledge.get("plants", [])]
        normalized = {name.lower(): name for name in names if name}

        for candidate in normalized:
            if candidate in text.lower():
                return normalized[candidate]
        return None

    def _generate_knowledge_based_response(self, question: str, prediction: dict[str, Any] | None = None) -> str:
        text = question.lower().strip()
        plant_name = None

        if not prediction:
            plant_name = self._find_known_plant(question)
            if plant_name is None:
                return "Upload a plant image and run the analysis so I can answer with the current plant context."

            plant_record = get_plant_info(plant_name)
            diseases = plant_record.get("diseases", [])
            disease_names = ", ".join(item.get("disease", "") for item in diseases[:3])
            return (
                f"{plant_name} is an agricultural crop commonly managed in field conditions. "
                f"The stored crop details include common issues such as: {disease_names}. "
                f"Ask about a specific problem such as symptoms, prevention, or care for {plant_name}."
            )

        plant_name = str(prediction.get("plant") or "Unknown plant")
        disease_name = str(prediction.get("disease") or "Unknown disease")
        plant_record = get_plant_info(plant_name)
        disease_record = get_disease_info(plant_name, disease_name)

        if "disease" in text and "what" in text:
            if prediction.get("healthy") is True:
                return f"The current image is most consistent with a healthy {plant_name} plant. I would continue routine care and monitoring rather than treating a disease."
            confidence = float(prediction.get("confidence", 0))
            threshold = float(prediction.get("threshold", 0.60))
            diagnosis_type = "possible match" if confidence < threshold else "likely diagnosis"
            description = disease_record.get("description", "")
            symptoms = disease_record.get("symptoms", [])
            symptom_summary = "; ".join(symptoms[:3])
            certainty_note = " Confidence is below the selected threshold, so confirm the match before treatment." if confidence < threshold else " This is a model prediction, not a confirmed diagnosis."
            return (
                f"{diagnosis_type}: {plant_name} {disease_name} ({confidence * 100:.1f}% confidence). "
                f"{description} Signs to check: {symptom_summary}.{certainty_note}"
            )

        if "symptom" in text:
            symptoms = disease_record.get("symptoms", []) or ["The disease symptoms are not clearly recorded in the local knowledge base."]
            return "Symptoms commonly associated with this issue include: " + "; ".join(symptoms)

        if "why" in text or "cause" in text or "happen" in text:
            contributing = disease_record.get("common contributing conditions", []) or ["Stress from moisture, poor airflow, or weak plant growth can increase disease risk."]
            return "The main contributing conditions are: " + "; ".join(contributing)

        if "prevent" in text:
            prevention = disease_record.get("prevention", []) or ["Use good sanitation, avoid overhead watering, and maintain plant spacing and airflow."]
            return "Prevention guidance: " + " ".join(prevention)

        if "grow" in text or "care" in text:
            care = disease_record.get("plant-care advice", []) or plant_record.get("description", "Keep the plant healthy with steady care and consistent monitoring.")
            return "Plant care guidance: " + " ".join(care if isinstance(care, list) else [care])

        if any(term in text for term in ("next", "do", "treat", "medicine", "pesticide", "fungicide", "spray", "chemical")):
            treatment = disease_record.get("treatment guidance", []) or ["Use a locally approved product for this crop and disease according to the product label and agricultural guidance."]
            return (
                f"Treatment guidance for {plant_name} {disease_name}: " + " ".join(treatment)
                + " The knowledge base does not verify a specific medicine name or dose. Use only products registered for this crop and disease in your area, follow the label, and confirm the diagnosis with a local agricultural expert. Tell me your country or region for more relevant guidance."
            )

        if "healthy" in text:
            if prediction.get("healthy") is True:
                return f"This image appears healthy overall for a {plant_name}. Continue light, watering, and monitoring routines and look for signs of stress before problems develop."
            return f"This current image is not consistent with a healthy plant. It is more likely to be {plant_name} {disease_name}, so I would check the affected leaves and improve prevention measures."

        if "monitor" in text or "often" in text:
            return "Monitor the plant closely every 2 to 3 days while symptoms are active. Check the newest leaves for spreading spots, wilting, or yellowing and document changes in photos."

        if "plant" in text and "name" in text:
            return f"The image appears to be associated with {plant_name}."

        confidence = float(prediction.get("confidence", 0))
        threshold = float(prediction.get("threshold", 0.60))
        diagnosis_type = "possible match" if confidence < threshold else "likely diagnosis"
        description = disease_record.get("description", "")
        symptoms = disease_record.get("symptoms", [])
        symptom_summary = "; ".join(symptoms[:3])
        return (
            f"Based on the image, the {diagnosis_type} is {plant_name} {disease_name} ({confidence * 100:.1f}% confidence). "
            f"{description} Signs to check: {symptom_summary}. Review the prevention and treatment guidance in the analysis."
        )

    def _generate_ollama_response(self, question: str, prediction: dict[str, Any] | None = None) -> str:
        if not prediction:
            return self._generate_knowledge_based_response(question, prediction)

        plant_name = str(prediction.get("plant") or "Unknown plant")
        disease_name = str(prediction.get("disease") or "Unknown disease")
        disease_info = get_disease_info(plant_name, disease_name)
        context = {
            "plant": plant_name,
            "disease": disease_name,
            "confidence": prediction.get("confidence", 0.0),
            "healthy": prediction.get("healthy", False),
            "info": disease_info,
        }

        prompt = (
            "You are a careful plant care assistant. The disease classifier prediction is only a prediction. "
            "Do not claim certainty. Use the retrieved plant information as the primary source. "
            f"Plant: {context['plant']}\n"
            f"Predicted disease: {context['disease']}\n"
            f"Confidence: {context['confidence'] * 100:.1f}%\n"
            f"Healthy status: {context['healthy']}\n"
            f"Disease info: {context['info']}\n"
            "Do not invent treatment products, dosages, or unsupported facts. "
            "If the confidence is low, recommend a clearer image and another angle. "
            "Ask the user to consult a qualified agricultural expert for serious or unusual plant problems. "
            f"User question: {question}"
        )

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        response = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", self._generate_knowledge_based_response(question, prediction))
