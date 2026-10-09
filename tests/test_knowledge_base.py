from services.knowledge_base import get_disease_info, get_plant_info, load_knowledge_base, search_knowledge


def test_knowledge_base_loads():
    data = load_knowledge_base()
    plants = {entry.get("plant") for entry in data.get("plants", [])}
    assert {"Tomato", "Potato", "Pepper", "Chilli"}.issubset(plants)


def test_get_plant_info_returns_record():
    plant = get_plant_info("Tomato")
    assert plant["plant"] == "Tomato"


def test_get_disease_info_returns_expected_data():
    info = get_disease_info("Tomato", "Early Blight")
    assert info["disease"] == "Early Blight"
    assert info["symptoms"]


def test_search_knowledge_finds_keyword():
    results = search_knowledge("late blight")
    assert len(results) > 0


def test_agri_crop_records_are_available():
    data = load_knowledge_base()
    plants = {entry.get("plant") for entry in data.get("plants", [])}
    assert {"Tomato", "Paddy", "Brinjal", "Coriander", "Wheat", "Maize", "Banana", "Groundnut", "Sugarcane", "Cotton", "Mango", "Coconut"}.issubset(plants)
