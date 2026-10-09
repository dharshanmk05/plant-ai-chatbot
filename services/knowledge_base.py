from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _knowledge_file_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "plant_knowledge.json"


def load_knowledge_base(path: str | None = None) -> dict[str, Any]:
    """Load the plant knowledge JSON file. Returns an empty structure if missing or invalid."""
    file_path = Path(path) if path else _knowledge_file_path()
    if not file_path.exists():
        return {"plants": []}

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {"plants": []}

    if isinstance(data, dict):
        return data if "plants" in data else {"plants": []}
    if isinstance(data, list):
        return {"plants": data}
    return {"plants": []}


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()


def get_plant_info(plant: str) -> dict[str, Any]:
    """Return the knowledge record for a plant or a safe fallback object."""
    plant_name = (plant or "").strip()
    knowledge = load_knowledge_base()
    for entry in knowledge.get("plants", []):
        if _normalize(entry.get("plant")) == _normalize(plant_name):
            return entry

    return {
        "plant": plant_name or "Unknown plant",
        "description": "No specific plant profile was found. Use the current prediction as a general guide and ask for a clearer image if needed.",
        "diseases": [],
    }


def get_disease_info(plant: str, disease: str) -> dict[str, Any]:
    """Return the matching disease record for a given plant and disease name."""
    plant_record = get_plant_info(plant)
    disease_name = (disease or "").strip()

    for disease_record in plant_record.get("diseases", []):
        if _normalize(disease_record.get("disease")) == _normalize(disease_name):
            return disease_record

    if disease_name.lower() in {"healthy", "no disease"}:
        return {
            "plant": plant,
            "disease": "Healthy",
            "description": "The plant appears healthy based on the current image.",
            "symptoms": ["No visible disease symptoms are evident in the provided image."],
            "common contributing conditions": ["Good plant health is maintained through consistent care and regular monitoring."],
            "prevention": ["Keep leaves dry during watering where practical and maintain airflow around the plant."],
            "plant-care advice": ["Provide appropriate light, water, and nutrition for the crop and continue regular observation."],
            "treatment guidance": ["No disease-treatment guidance is needed for a healthy plant."],
            "severity": "Low",
            "notes": ["Observation and good cultural practice remain important."]
        }

    return {
        "plant": plant,
        "disease": disease_name or "Unknown disease",
        "description": "No detailed disease record is available for this case. Use a clearer image and check the plant for visible symptoms.",
        "symptoms": ["Symptoms could not be matched to a known disease entry."],
        "common contributing conditions": ["Poor image quality, stress, or environmental conditions may make diagnosis difficult."],
        "prevention": ["Use a clearer image and make sure the leaf is well-lit and in focus."],
        "plant-care advice": ["Monitor plant health regularly and avoid stress from water imbalance or poor airflow."],
        "treatment guidance": ["Use a locally approved product for this crop and disease according to the product label and agricultural guidance."],
        "severity": "Unknown",
        "notes": ["This is a general fallback answer when the disease record is missing."],
    }


def search_knowledge(query: str) -> list[dict[str, Any]]:
    """Search plant and disease knowledge text fields by keyword."""
    if not query or not query.strip():
        return []

    term = _normalize(query)
    matches: list[dict[str, Any]] = []
    knowledge = load_knowledge_base()

    for plant_entry in knowledge.get("plants", []):
        plant_name = str(plant_entry.get("plant", ""))
        description = str(plant_entry.get("description", ""))
        searchable_text = " ".join(
            [plant_name, description, *[str(item) for item in plant_entry.get("diseases", [])]]
        ).lower()
        if term in searchable_text:
            matches.append(plant_entry)
            continue

        for disease_entry in plant_entry.get("diseases", []):
            disease_text = " ".join(
                [
                    str(disease_entry.get("disease", "")),
                    str(disease_entry.get("description", "")),
                    " ".join(str(item) for item in disease_entry.get("symptoms", [])),
                    " ".join(str(item) for item in disease_entry.get("common contributing conditions", [])),
                    " ".join(str(item) for item in disease_entry.get("prevention", [])),
                    " ".join(str(item) for item in disease_entry.get("plant-care advice", [])),
                ]
            ).lower()
            if term in disease_text:
                matches.append(disease_entry)
    return matches
