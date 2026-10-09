MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
CONFIDENCE_THRESHOLD = 0.60
CACHE_DIR = "./.cache/huggingface"
DISEASE_MODEL_PLANTS = [
    "Apple",
    "Blueberry",
    "Cherry",
    "Maize",
    "Grapevine",
    "Orange",
    "Peach",
    "Pepper",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato",
]
SUPPORTED_PLANTS = [
    "Tomato",
    "Potato",
    "Pepper",
    "Chilli",
    "Paddy",
    "Brinjal",
    "Coriander",
    "Onion",
    "Okra",
    "Wheat",
    "Maize",
    "Banana",
    "Groundnut",
    "Sugarcane",
    "Cotton",
    "Cucumber",
    "Cabbage",
    "Cauliflower",
    "Mango",
    "Grapevine",
    "Soybean",
    "Sunflower",
    "Coconut",
]
MAX_IMAGE_DIMENSION = 224

LOCAL_OLLAMA_URL = "http://localhost:11434"
LOCAL_OLLAMA_MODEL = "llama3.2"
