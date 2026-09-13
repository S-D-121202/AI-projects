import torch
import torch.nn.functional as F
import re
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image
from gtts import gTTS
import os
from langchain_google_genai import ChatGoogleGenerativeAI

from transformers import AutoProcessor, AutoTokenizer

from VLM import SCOLD
from SCOLD_explainability_test_version import (
    SCOLDAttentionExtractor,
    SCOLDGradCAM,
    SCOLDVisualizer
)

os.environ["GOOGLE_API_KEY"] = "AQ.Ab8RN6JoGg0N0EoZKRqv4akgEJ8kNfNmLmBnOScxqpdoHa7q-A"

chat_model = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=1,
    thinking_level="low",
    request_timeout=30,
    max_retries=0
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon=None,
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_PATH = "./saved_clip_model"

SCOLD_WEIGHTS_PATH = (
    "./finetuned_scold_leafnet_ver2/scold_weights.pth"
)

CLASS_SENTENCES_PATH = "class_sentences.npy"

OLLAMA_MODEL = "llama3"


# ============================================================
# LOAD SCOLD MODEL
# ============================================================

@st.cache_resource
def load_scold_model():

    model = SCOLD(
        base_model_name=BASE_MODEL_PATH
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            SCOLD_WEIGHTS_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True
    )

    classes = np.load(
        CLASS_SENTENCES_PATH,
        allow_pickle=True
    ).tolist()

    return model, processor, tokenizer, classes


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner("Loading backend model..."):

    model, processor, tokenizer, classes = (
        load_scold_model()
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "Plant Disease Detection and Explainability"
)

st.markdown(
    """
    Upload a plant leaf image and click **Analyze Image**.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Configuration")

    st.write(
        f"Device: `{DEVICE}`"
    )

    st.write(
        f"LLM: Gemini-3.7-flash"
    )

    st.divider()

    st.info(
        "Upload a clear image of a plant leaf "
        "for better classification and localization."
    )


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "predicted_prompt" not in st.session_state:
    st.session_state.predicted_prompt = None

if "llm_output" not in st.session_state:
    st.session_state.llm_output = None

if "explainability_data" not in st.session_state:
    st.session_state.explainability_data = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_language_sections(text):

    english_match = re.search(
        r"\*\*English\*\*(.*?)(?=\*\*Bengali\*\*|\Z)",
        text,
        re.DOTALL | re.IGNORECASE
    )

    bengali_match = re.search(
        r"\*\*Bengali\*\*(.*?)(?=\*\*Hindi\*\*|\Z)",
        text,
        re.DOTALL | re.IGNORECASE
    )

    hindi_match = re.search(
        r"\*\*Hindi\*\*(.*)",
        text,
        re.DOTALL | re.IGNORECASE
    )

    english = (
        english_match.group(1).strip()
        if english_match
        else ""
    )

    bengali = (
        bengali_match.group(1).strip()
        if bengali_match
        else ""
    )

    hindi = (
        hindi_match.group(1).strip()
        if hindi_match
        else ""
    )

    return english, bengali, hindi


def clean_text_for_speech(text):

    text = re.sub(
        r"\*+",
        "",
        text
    )

    text = re.sub(
        r"^\s*[\d\.\-\*]+\s*",
        "",
        text,
        flags=re.MULTILINE
    )

    text = re.sub(
        r"^(Input Data|Remedial Measures)\s*:\s*",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE
    )

    text = re.sub(
        r"\n+",
        ". ",
        text
    )

    return text.strip()


@st.cache_data
def generate_tts_bytes(text, lang_code):

    if not text:
        return None

    audio_buffer = BytesIO()

    tts = gTTS(
        text=text,
        lang=lang_code,
        slow=False
    )

    tts.write_to_fp(audio_buffer)

    audio_buffer.seek(0)

    return audio_buffer.getvalue()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# DISPLAY IMAGE AND INFERENCE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Input Image")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    st.divider()

    analyze_button = st.button(
        "Analyze Image",
        type="primary",
        use_container_width=True
    )

    # ========================================================
    # RUN ANALYSIS AND STORE IN SESSION STATE
    # ========================================================

    if analyze_button:

        # Clear GPU Memory before execution
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with st.spinner("Processing image and running SCOLD classification..."):

            pixel_values = processor(
                images=image,
                return_tensors="pt"
            )["pixel_values"].to(DEVICE)

            text_inputs = tokenizer(
                classes,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():

                text_embeds = model.encode_text(
                    text_inputs.input_ids,
                    text_inputs.attention_mask
                )

                text_embeds = F.normalize(
                    text_embeds, p=2, dim=-1
                )

                image_embed = model.encode_image(
                    pixel_values
                )

                image_embed = F.normalize(
                    image_embed, p=2, dim=-1
                )

                similarity = torch.matmul(
                    image_embed,
                    text_embeds.t()
                )

                logit_scale = model.logit_scale.exp()

                logits = logit_scale * similarity

                probabilities = F.softmax(
                    logits, dim=-1
                ).squeeze(0)

                top_scores, top_indices = torch.topk(
                    probabilities, k=1
                )

                predicted_idx = top_indices[0].item()

                st.session_state.predicted_prompt = classes[predicted_idx]

        # Ollama LLM Explanation
        gemini_prompt = f"""
You are an agricultural plant disease assistant.

The image classification model predicted the following
plant disease/state:

Input data: {st.session_state.predicted_prompt}

Your task:

1. Explain the predicted plant disease/state in simple language.
2. If the plant is diseased, provide practical remedial measures.
3. If the plant appears healthy, clearly state that no disease
   treatment is required and provide brief preventive advice.
4. Do not invent a different disease from the given input data.
5. Provide the information in English, Bengali, and Hindi.
6. Keep the response concise.
7. Do not repeat any sentence or phrase.

Output EXACTLY in this format:

**English**
Input Data: ...
Remedial Measures: ...

**Bengali**
Input Data: ...
Remedial Measures: ...

**Hindi**
Input Data: ...
Remedial Measures: ...
"""

        with st.spinner("Generating explanation..."):
            response = chat_model.invoke(gemini_prompt)
            st.session_state.llm_output = response.content[0]['text']
            
        # Explainability Generation
        with st.spinner("Generating Grad-CAM explainability maps..."):

            image_np = np.array(image)

            attn_extractor = SCOLDAttentionExtractor(model)
            grad_cam = SCOLDGradCAM(model)

            rollout_map = attn_extractor.get_attention_rollout(pixel_values)

            gradcam_map = grad_cam.generate_heatmap(
                pixel_values=pixel_values,
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask'],
                target_class_idx=top_indices[0].item()
            )

            blended_img, heatmap_2d = SCOLDVisualizer.overlay_heatmap(
                image_np, rollout_map
            )

            bbox_img, boxes = SCOLDVisualizer.draw_disease_bounding_boxes(
                blended_img, heatmap_2d
            )

            st.session_state.explainability_data = {
                "heatmap_2d": heatmap_2d,
                "blended_img": blended_img,
                "bbox_img": bbox_img
            }

        # Clear GPU Memory after execution
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # ========================================================
    # INTERACTIVE SESSION CONTROLS (POST-ANALYSIS)
    # ========================================================

    if st.session_state.predicted_prompt is not None:

        st.subheader("Disease Classification")
        st.success(f"Prediction: {st.session_state.predicted_prompt}")

        st.divider()

        # Dynamic Controls for output customization
        st.subheader("Display Options")

        col_opt1, col_opt2, col_opt3 = st.columns(3)

        with col_opt1:
            selected_lang = st.selectbox(
                "Select Language",
                ["All", "English", "Bengali", "Hindi"],
                key="lang_select"
            )

        with col_opt2:
            show_audio = st.checkbox(
                "Show Audio Controls",
                value=True,
                key="audio_check"
            )

        with col_opt3:
            show_explainability = st.checkbox(
                "Show Explainability Visuals",
                value=True,
                key="explain_check"
            )

        # Parse stored text output
        english_raw, bengali_raw, hindi_raw = extract_language_sections(
            st.session_state.llm_output
        )

        english_speech = clean_text_for_speech(english_raw)
        bengali_speech = clean_text_for_speech(bengali_raw)
        hindi_speech = clean_text_for_speech(hindi_raw)

        # Display Text
        st.divider()
        st.subheader("Detailed Explanation")

        if selected_lang == "English":
            st.markdown("### English")
            st.write(english_raw if english_raw else "English output not detected.")
        elif selected_lang == "Bengali":
            st.markdown("### Bengali")
            st.write(bengali_raw if bengali_raw else "Bengali output not detected.")
        elif selected_lang == "Hindi":
            st.markdown("### Hindi")
            st.write(hindi_raw if hindi_raw else "Hindi output not detected.")
        else:
            lang_col1, lang_col2, lang_col3 = st.columns(3)
            with lang_col1:
                st.markdown("### English")
                st.write(english_raw if english_raw else "English output not detected.")
            with lang_col2:
                st.markdown("### Bengali")
                st.write(bengali_raw if bengali_raw else "Bengali output not detected.")
            with lang_col3:
                st.markdown("### Hindi")
                st.write(hindi_raw if hindi_raw else "Hindi output not detected.")

        # Display Audio
        if show_audio:
            st.divider()
            st.subheader("Multilingual Audio")

            if selected_lang == "English" and english_speech:
                st.audio(generate_tts_bytes(english_speech, "en"), format="audio/mp3")
            elif selected_lang == "Bengali" and bengali_speech:
                st.audio(generate_tts_bytes(bengali_speech, "hi"), format="audio/mp3")
            elif selected_lang == "Hindi" and hindi_speech:
                st.audio(generate_tts_bytes(hindi_speech, "hi"), format="audio/mp3")
            else:
                aud_col1, aud_col2, aud_col3 = st.columns(3)
                with aud_col1:
                    if english_speech:
                        st.markdown("**English Audio**")
                        st.audio(generate_tts_bytes(english_speech, "en"), format="audio/mp3")
                with aud_col2:
                    if bengali_speech:
                        st.markdown("**Bengali Audio**")
                        st.audio(generate_tts_bytes(bengali_speech, "bn"), format="audio/mp3")
                with aud_col3:
                    if hindi_speech:
                        st.markdown("**Hindi Audio**")
                        st.audio(generate_tts_bytes(hindi_speech, "hi"), format="audio/mp3")

        # Display Explainability Visuals
        if show_explainability and st.session_state.explainability_data:
            st.divider()
            st.subheader("Model Explainability")

            exp_data = st.session_state.explainability_data

            exp_col1, exp_col2, exp_col3 = st.columns(3)

            with exp_col1:
                st.image(
                    exp_data["heatmap_2d"],
                    caption="2D Heatmap Intensity",
                    use_container_width=True,
                    clamp=True
                )

            with exp_col2:
                st.image(
                    exp_data["blended_img"],
                    caption="Blended Image Overlay",
                    use_container_width=True
                )

            with exp_col3:
                st.image(
                    exp_data["bbox_img"],
                    caption="Disease Bounding Boxes",
                    use_container_width=True
                )
