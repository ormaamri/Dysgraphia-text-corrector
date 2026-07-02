import streamlit as st
import easyocr
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO  
import os
import shutil

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
    llm_model = genai.GenerativeModel('gemini-3.5-flash')

    return yolo_model, reader, llm_model

yolo_model, reader, llm_model = load_models()

# ==========================================
# PROCESSING FUNCTIONS
# ==========================================
def segment_image(image, model):
    if model is None:
        return [{"crop": image, "label": "Unknown"}]
    
    results = model(image)
    segments = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label_name = model.names[cls_id]
            
            crop = image[max(0, y1):y2, max(0, x1):x2]
            
            if crop.size > 0:
                segments.append({
                    "crop": crop,
                    "label": label_name
                })
                
    if not segments:
        return [{"crop": image, "label": "Unknown"}]
        
    return segments

def analyze_and_correct_with_gemini(text, model):
    """
    Asks Gemini to both classify the text quality and perform corrections if needed.
    """
    prompt = f"""
    You are an AI analyzing OCR text extracted from handwriting to detect dysgraphia or dyslexia.
    Analyze this text: "{text}"

    Determine if this text is normal, clear English or if it contains severe dysgraphic errors, structural confusions, or scrambled words that need reconstruction.

    Respond in exactly this format:
    STATUS: [Write either NORMAL or DYSGRAPHIC]
    TEXT: [If NORMAL, repeat the input text exactly. If DYSGRAPHIC, perfectly reconstruct the intended English sentence.]
    """
    try:
        response = model.generate_content(prompt)
        lines = response.text.strip().split('\n')
        status = "NORMAL"
        result_text = text
        
        for line in lines:
            if line.upper().startswith("STATUS:"):
                status = line.split(":")[1].strip().upper()
            elif line.upper().startswith("TEXT:"):
                result_text = line.split(":")[1].strip()
                
        return status, result_text
    except Exception as e:
        return "NORMAL", f"Error connecting to Gemini: {e}"

def correct_text_only(garbled_text, model):
    """Fallback standard prompt for direct YOLO classifications"""
    prompt = f"""
    You are an expert at deciphering severely dyslexic and dysgraphic handwriting.
    Read this garbled OCR output and perfectly reconstruct the intended English sentence.
    Only output the corrected sentence, nothing else.

    Garbled Text: {garbled_text}
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error connecting to Gemini: {e}"

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
            with st.spinner("Analyzing handwriting with YOLO and extracting text..."):
                image_segments = segment_image(image_np, yolo_model)
                
                if not image_segments:
                    st.info("No text segments detected.")
                else:
                    for i, segment in enumerate(image_segments):
                        crop = segment["crop"]
                        label = segment["label"]
                        
                        st.image(crop, caption=f"Processing Segment {i+1}", width=150)
                        
                        raw_results = reader.readtext(crop, detail=0, paragraph=True)
                        
                        if raw_results:
                            combined_text = " ".join(raw_results)
                            
                            # CASE 1: YOLO confidently says it is Normal handwriting
                            if label.lower() == "normal":
                                st.markdown(f"""
                                <div class="normal-card">
                                    <span class="label-badge" style="background:#E8F5E9; color:#2E7D32;">✓ NORMAL TEXT DETECTED</span>
                                    <div class="garbled-text"><b>OCR Extraction:</b> {combined_text}</div>
                                    <div class="normal-text">ℹ️ The text in this image is not dysgraphic text, and does not need to be translated.</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # CASE 2: YOLO couldn't find bounding boxes (Unknown). Let Gemini decide!
                            elif label.lower() == "unknown":
                                status, verified_text = analyze_and_correct_with_gemini(combined_text, llm_model)
                                
                                if status == "NORMAL":
                                    st.markdown(f"""
                                    <div class="normal-card">
                                        <span class="label-badge" style="background:#E8F5E9; color:#2E7D32;">✓ NO HANDWRITING DETECTED</span>
                                        <div class="garbled-text"><b>OCR Extraction:</b> {combined_text}</div>
                                        <div class="normal-text">ℹ️ The text in this image is not dysgraphic text, and does not need to be translated.</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.info(f"Dysgraphic text has been detected, proceeding to translate...")
                                    st.markdown(f"""
                                    <div class="result-card">
                                        <span class="label-badge" style="background:#F3E5F5; color:#6A1B9A;">⚠️ DYSGRAPHIA DETECTED</span>
                                        <div class="garbled-text"><b>Raw OCR Extraction:</b> {combined_text}</div>
                                        <div class="corrected-text">✨ <b>Translated:</b> {verified_text}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # CASE 3: YOLO clearly catches a dysgraphic class (Reversal, Corrected, etc.)
                            else:
                                st.info(f"Dysgraphic text has been detected, proceeding to translate...")
                                corrected_line = correct_text_only(combined_text, llm_model)
                                
                                st.markdown(f"""
                                <div class="result-card">
                                    <span class="label-badge" style="background:#F3E5F5; color:#6A1B9A;">⚠️ {label.upper()} TEXT DETECTED</span>
                                    <div class="garbled-text"><b>Raw OCR Extraction:</b> {combined_text}</div>
                                    <div class="corrected-text">✨ <b>Translated:</b> {corrected_line}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.write("No readable text found in this segment.")