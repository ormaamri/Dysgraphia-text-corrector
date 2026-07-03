import streamlit as st
import easyocr
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import shutil
import time
import re

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Dysgraphia Translator",
    page_icon="✍️",
    layout="wide"
)

st.markdown("""
    <style>
    .result-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(106, 27, 154, 0.15);
        border-left: 6px solid #6A1B9A;
        margin-bottom: 15px;
        color: #333333;
    }
    .normal-card {
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border-left: 6px solid #4CAF50; 
        margin-bottom: 15px;
        color: #333333;
    }
    .garbled-text {
        color: #555555;
        font-style: italic;
        margin-bottom: 10px;
        font-size: 0.95em;
    }
    .corrected-text {
        color: #4A148C;
        font-weight: 600;
        font-size: 1.15em;
        margin-top: 10px;
    }
    .normal-text {
        color: #2E7D32;
        font-weight: 600;
        font-size: 1.1em;
    }
    .label-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        margin-bottom: 12px;
        text-transform: uppercase;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# MODEL INITIALIZATION (CACHED)
# ==========================================
@st.cache_resource
def load_models():
    original_file = 'YOLO.keras'
    temp_file = 'temp_model.pt'
    
    if os.path.exists(original_file) and not os.path.exists(temp_file):
        shutil.copy(original_file, temp_file)
        
    try:
        yolo_model = YOLO(temp_file if os.path.exists(temp_file) else original_file)
    except Exception as e:
        yolo_model = None
        st.warning(f"YOLO Model Warning: Could not load {original_file}. Make sure it is in the same folder as app.py!") 

    reader = easyocr.Reader(['en'], gpu=False)

    api_key = st.secrets.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        
    # FIX: Ensure we use a model that supports vision (1.5-flash)
    llm_model = genai.GenerativeModel('gemini-3.5-flash')

    return yolo_model, reader, llm_model

yolo_model, reader, llm_model = load_models()

# ==========================================
# PROCESSING FUNCTIONS
# ==========================================
def analyze_image_and_correct(image_file, model):
    """
    Uses Gemini Vision to visually inspect the handwriting for dysgraphia
    and extract/correct the text in one pass.
    """
    prompt = """
    You are an expert in detecting dysgraphia and a professional proofreader. 
    Analyze the provided image of handwriting or printed text.
    
    Task:
    1. Visually inspect the image for signs of dysgraphia. Look for:
       - Erratic or absent spacing between words.
       - Irregular or inconsistent letter sizing.
       - Letters floating off the baseline or overlapping.
       - Frequent cross-outs, corrections, or reversed letters.
    2. If the writing is standard, neat, or printed perfectly, do not flag it.
    3. Read the text. If there are dysgraphic mistakes or spelling errors, reconstruct the intended message into clear, grammatically correct English. If it's normal, just extract it exactly as written.

    Respond EXACTLY with these two lines. Do not use Markdown, bolding, or extra commentary.
    DYSGRAPHIC: [YES if you visually detect dysgraphia, NO if the writing is neat/normal]
    CORRECTED: [Provide the clean, correct English text here.]
    """
    
    try:
        # Pass both the text prompt and the image object directly to Gemini
        response = model.generate_content([prompt, image_file])
        response_text = response.text.strip()
        
        is_dysgraphic = False
        corrected_text = ""
        
        # Parse the output
        lines = response_text.split('\n')
        for i, line in enumerate(lines):
            clean_line = line.strip().upper().replace('*', '')
            
            if clean_line.startswith("DYSGRAPHIC:"):
                if "YES" in clean_line:
                    is_dysgraphic = True
                else:
                    is_dysgraphic = False
            elif clean_line.startswith("CORRECTED:"):
                corrected_text = "\n".join(lines[i:]).strip()
                corrected_text = re.sub(r'(?i)^\*?CORRECTED:\*?\s*', '', corrected_text).strip()
                break
                
        return is_dysgraphic, corrected_text

    except Exception as e:
        error_msg = f"[API Error: {str(e)}]\nPlease check your API Key or connection."
        return True, error_msg

# ==========================================
# UI LAYOUT
# ==========================================
st.title("✍️ Dysgraphia Text Translator")
st.markdown("Upload an image of handwriting below. The AI pipeline will classify the text, extract it, and translate dysgraphic elements into clear English while leaving normal text intact.")

uploaded_file = st.file_uploader("Upload Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    st.divider() 
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Analysis & Results")
        
        if not st.secrets.get("GEMINI_API_KEY"):
            st.error("Gemini API Key missing. Please check your secrets.toml file.")
        else:
            with st.spinner("Analyzing image and extracting text..."):
                
                # Run EasyOCR on the FULL image to preserve exact reading order (for displaying raw output)
                raw_results = reader.readtext(image_np, detail=0, paragraph=True)
                full_text = " ".join(raw_results).strip()
                
                if not full_text:
                    st.warning("Could not extract any readable text from the image.")
                else:
                    # FIX: Pass the PIL Image and use the correct function name
                    is_dysgraphic, final_text = analyze_image_and_correct(image, llm_model)
                    
                    if is_dysgraphic:
                        st.error("🚨 **Detection Result:** Dysgraphia Detected")
                        
                        st.markdown(f"""
                        <div class="result-card">
                            <span class="label-badge" style="background:#F3E5F5; color:#6A1B9A;">⚠️ DYSGRAPHIA DETECTED</span>
                            <div class="garbled-text"><b>Raw OCR Output:</b><br>{full_text}</div>
                            <div class="corrected-text">✨ <b>Translated Paragraph:</b><br>{final_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success("✅ **Detection Result:** Normal Handwriting / Text")
                        
                        # FIX: Display `final_text` instead of `full_text` to show the clean reading
                        st.markdown(f"""
                        <div class="normal-card">
                            <span class="label-badge" style="background:#E8F5E9; color:#2E7D32;">✓ NORMAL TEXT</span>
                            <div class="normal-text">{final_text}</div>
                        </div>
                        """, unsafe_allow_html=True)