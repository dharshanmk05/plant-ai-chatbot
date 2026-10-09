from __future__ import annotations

import streamlit as st

from model.model_config import CONFIDENCE_THRESHOLD, DISEASE_MODEL_PLANTS, MODEL_NAME, SUPPORTED_PLANTS
from services.chatbot import PlantCareChatbot
from services.disease_predictor import PlantDiseasePredictor
from services.image_utils import basic_image_quality_check, load_image
from services.knowledge_base import get_disease_info

st.set_page_config(page_title="🌱 Plant AI Assistant", layout="wide")

st.title("🌱 Plant AI Assistant")
st.caption("AI-based plant disease identification and intelligent plant care")

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [{"role": "assistant", "content": "Upload a plant image and click Analyze Plant to begin."}]

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "Knowledge Base"

with st.sidebar:
    st.header("About")
    st.write(
        "This app helps identify likely plant diseases from a leaf image and gives support for plant care, prevention, and follow-up questions."
    )

    st.subheader("Supported Plants")
    st.write(", ".join(SUPPORTED_PLANTS))

    st.subheader("Model Information")
    st.write(f"Hugging Face model: {MODEL_NAME}")
    st.write("The disease model recognizes common PlantVillage leaf classes for: " + ", ".join(DISEASE_MODEL_PLANTS) + ".")
    st.write("Other crops are available in the knowledge base, but this image model does not classify their diseases. Field results can differ from controlled dataset images.")
    st.write("The model is downloaded when you first click Analyze Plant and cached locally afterward.")

    st.subheader("Confidence Threshold")
    st.session_state.confidence_threshold = st.slider(
        "Minimum confidence for diagnosis",
        min_value=0.40,
        max_value=0.95,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
    )

    st.subheader("Chat Mode")
    st.session_state.chat_mode = st.radio(
        "Choose chatbot mode",
        ["Knowledge Base", "Local Ollama (optional)"],
        index=0,
    )

    if st.button("Clear Chat"):
        st.session_state.chat_messages = [{"role": "assistant", "content": "Upload a plant image and click Analyze Plant to begin."}]

    st.write("\n")
    st.write("The app works without Ollama. If the local model is unavailable, it automatically falls back to the knowledge-based chatbot.")

uploaded_file = st.file_uploader("Upload a clear image of a plant leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded plant image", use_container_width=True)

    if st.button("Analyze Plant", type="primary"):
        try:
            image = load_image(uploaded_file)
            quality = basic_image_quality_check(image)
            if not quality["is_valid"]:
                st.warning(quality["message"])
                st.session_state.latest_prediction = None
            else:
                with st.spinner("Loading the model and analyzing the image..."):
                    predictor = PlantDiseasePredictor(
                        model_name=MODEL_NAME,
                        threshold=st.session_state.confidence_threshold,
                    )
                    prediction = predictor.analyze(image)
                    st.session_state.latest_prediction = prediction

                    if prediction.get("healthy") is True:
                        assistant_message = (
                            f"Your image appears healthy for {prediction['plant']} with a confidence of {prediction['confidence'] * 100:.1f}%."
                        )
                    else:
                        disease_info = get_disease_info(prediction["plant"], prediction["disease"])
                        confidence_percent = prediction["confidence"] * 100
                        diagnosis_type = "Possible match" if prediction["confidence"] < st.session_state.confidence_threshold else "Likely diagnosis"
                        description = disease_info.get("description", "")
                        symptoms = disease_info.get("symptoms", [])
                        symptom_summary = "; ".join(symptoms[:3])
                        assistant_message = (
                            f"{diagnosis_type}: {prediction['plant']} {prediction['disease']} ({confidence_percent:.1f}% confidence). "
                            f"{description} Signs to check: {symptom_summary}."
                        )
                        if prediction["confidence"] < st.session_state.confidence_threshold:
                            assistant_message += " Confidence is below the selected threshold; confirm the diagnosis before treatment."

                    if not any(message.get("content") == assistant_message for message in st.session_state.chat_messages):
                        st.session_state.chat_messages.append({"role": "assistant", "content": assistant_message})

        except ValueError as exc:
            st.warning(str(exc))
            st.session_state.latest_prediction = None
        except RuntimeError as exc:
            st.error(str(exc))
            st.session_state.latest_prediction = None
        except Exception as exc:  # pragma: no cover - final catch for unexpected user-facing issues
            st.error("The application could not complete the analysis. Please try another image or retry in a moment.")
            st.session_state.latest_prediction = None

if st.session_state.latest_prediction:
    prediction = st.session_state.latest_prediction
    confidence_percent = prediction["confidence"] * 100
    plant = prediction["plant"]
    disease = prediction["disease"]
    disease_info = get_disease_info(plant, disease)

    st.markdown("---")
    st.subheader("Plant Analysis")
    metrics = st.columns(4)
    metrics[0].metric("Plant", plant)
    metrics[1].metric("Predicted Condition", disease)
    metrics[2].metric("Confidence", f"{confidence_percent:.1f}%")
    metrics[3].metric("Status", prediction.get("status", "Unknown"))

    if prediction["healthy"]:
        st.success("Status: Healthy")
        st.write("The image is consistent with a healthy plant. Continue routine care and monitoring.")
    else:
        description = disease_info.get("description", "")
        symptoms = disease_info.get("symptoms", [])
        if prediction["confidence"] < st.session_state.confidence_threshold:
            st.warning(f"Possible match: {plant} {disease} ({confidence_percent:.1f}% confidence), below the selected threshold.")
            st.write("Please verify the symptoms before taking action. Upload a closer, well-lit image that shows both affected and healthy tissue.")
        else:
            st.info(f"Likely diagnosis: {plant} {disease} ({confidence_percent:.1f}% confidence). This is a model prediction, not a confirmed diagnosis.")

        if description:
            st.write(description)
        if symptoms:
            st.markdown("**Affected parts and signs to check**")
            for item in symptoms:
                st.write(f"- {item}")
        if prediction["confidence"] < st.session_state.confidence_threshold:
            st.caption("Do not choose or apply medicine based only on this low-confidence image result.")

    if prediction["healthy"]:
        with st.expander("Basic Care Suggestions"):
            st.write("- Use consistent, moderate watering and avoid stress from overwatering or underwatering.")
            st.write("- Provide enough light for the crop and maintain healthy airflow around the plant.")
            st.write("- Check leaves regularly for any changes in color, texture, or growth pattern.")
            st.write("- Keep a simple monitoring routine for new pests, nutrient problems, or leaf stress.")
    elif prediction["confidence"] >= st.session_state.confidence_threshold:
        with st.expander("Possible Causes"):
            causes = disease_info.get("common contributing conditions", [])
            if causes:
                for item in causes:
                    st.write(f"- {item}")
            else:
                st.write("- No specific cause information was matched for this case.")

        with st.expander("Precautions"):
            prevention = disease_info.get("prevention", [])
            if prevention:
                for item in prevention:
                    st.write(f"- {item}")
            else:
                st.write("- Follow good crop hygiene and avoid wet foliage when possible.")

        with st.expander("Plant Care"):
            care = disease_info.get("plant-care advice", [])
            if care:
                for item in care:
                    st.write(f"- {item}")
            else:
                st.write("- Continue routine plant monitoring and maintain healthy crop conditions.")

        with st.expander("Treatment / Medicine Guidance"):
            treatment = disease_info.get("treatment guidance", [])
            if treatment:
                for item in treatment:
                    st.write(f"- {item}")
            else:
                st.write("- Use locally approved products according to the label and local agricultural guidance.")
            st.caption("The knowledge base does not verify a specific medicine name or dose. Ask for a crop- and location-approved product, follow its label and harvest interval, and do not treat from an image prediction alone.")

    st.write("\n")
    st.write("Top model predictions:")
    for item in prediction.get("top_predictions", [])[:3]:
        label = item.get("label", "Unknown")
        score = item.get("score", 0.0)
        st.write(f"- {label}: {score * 100:.1f}%")

st.markdown("---")
st.subheader("Plant Care Chat")

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about this plant health or care")
if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    bot = PlantCareChatbot()
    mode = "local_llm" if st.session_state.chat_mode == "Local Ollama (optional)" else "knowledge"
    answer = bot.generate_response(prompt, prediction=st.session_state.latest_prediction, mode=mode)

    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
