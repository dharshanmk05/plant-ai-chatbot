# AI-Based Plant Disease Detection and Intelligent Plant Care Chatbot

A beginner-friendly Streamlit project for plant image analysis and plant-care chat support on a normal Windows laptop.

## Features

- Upload a plant or leaf image
- Classify the image with a Hugging Face image-classification model
- Show predicted plant, condition, confidence, and health status
- Use a local knowledge base for disease symptoms, causes, and care guidance
- Chat about the latest analysis using a rule-based knowledge chatbot
- Optional local Ollama-based chat when available
- Graceful handling of poor or unclear images

## Project folders

- app.py: Streamlit app
- model/model_config.py: model settings
- services/: predictor, chatbot, knowledge, and image utilities
- data/plant_knowledge.json: structured plant and disease knowledge
- tests/: small validation tests

## Run locally

1. Open a terminal in the project root.
2. Create a virtual environment.
3. Install dependencies:

   pip install -r requirements.txt

4. Start the app:

   streamlit run app.py

## Notes

- The app attempts to download the default Hugging Face image-classification model on first use when internet access is available.
- If the model is unavailable, a friendly error is shown and the app continues to function.
- The project is designed for educational and demonstration use.
