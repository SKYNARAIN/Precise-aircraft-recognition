import streamlit as st
import tempfile
import os
import cv2
import json
import numpy as np
from google import genai
from google.genai.errors import APIError
from dotenv import load_dotenv

# -----------------------------
# Load Gemini API Key
# -----------------------------
load_dotenv()
GEMINI_API_KEY = "Enter key here"
if not GEMINI_API_KEY:
    st.error("⚠️ Please set GEMINI_API_KEY in your .env file.")
    st.stop()

# Initialize Gemini client
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"❌ Failed to initialize Gemini client: {e}")
    st.stop()

MODEL_NAME = "gemini-2.5-flash"
MAX_RETRIES = 5

st.set_page_config(page_title="Aircraft Video Analyzer", layout="wide")
st.title("✈️ Precise Aircraft Recognition")

# -----------------------------
# Functions
# -----------------------------
def run_gemini_analysis(video_path):
    """Send video to Gemini and return structured JSON result (silent on failure)"""
    try:
        video_file = client.files.upload(file=video_path)
        prompt_parts = [video_file, "Identify the airline and airplane model in this video."]
        system_instruction = (
            "You are an expert aviation analyst. Analyze the video and identify airline and aircraft model. "
            "Return structured JSON with 'airline', 'model', 'analysis'."
        )
        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt_parts,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json"
                    ),
                )
                return json.loads(response.text)
            except Exception:
                # silently retry
                continue
        # If all retries fail, return clean fallback
        return {"airline": "Unknown", "model": "Unknown", "analysis": "Analysis failed."}
    except Exception:
        # Silent fail
        return {"airline": "Unknown", "model": "Unknown", "analysis": "Analysis failed."}

def process_video(input_path, output_path, analysis):
    """Overlay analysis on video frames and save"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error(f"❌ Could not open video: {input_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 for browser playback
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    airline_text = f"Airline: {analysis.get('airline','N/A')}"
    model_text = f"Model: {analysis.get('model','N/A')}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = width / 1200
    thickness = max(1, int(2 * scale))
    color = (0, 255, 0)
    margin = int(width * 0.03)

    (tw1, th1), _ = cv2.getTextSize(airline_text, font, scale, thickness)
    (tw2, th2), _ = cv2.getTextSize(model_text, font, scale, thickness)
    box_height = th1 + th2 + margin*3
    x, y = margin, margin

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Draw overlay box
        cv2.rectangle(frame, (x, y - margin), (x + max(tw1, tw2) + margin*2, y + box_height), (255,255,255), cv2.FILLED)
        cv2.putText(frame, airline_text, (x+10, y+th1), font, scale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, model_text, (x+10, y+th1+th2+10), font, scale, color, thickness, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()
    return output_path

# -----------------------------
# Main UI
# -----------------------------
uploaded_file = st.file_uploader("Upload an MP4 video", type=["mp4"])

if uploaded_file:
    with st.spinner("📤 Uploading and analyzing video, please wait..."):
        # Save upload to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
            tmp_input.write(uploaded_file.read())
            input_path = tmp_input.name

        # Run Gemini analysis
        analysis_result = run_gemini_analysis(input_path)

        st.subheader("✈️ Analysis Result")
        st.json(analysis_result)

        # Process video
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        processed_path = process_video(input_path, output_file, analysis_result)

    if processed_path:
        st.subheader("🎬 Processed Video")
        st.video(processed_path, start_time=0)

        st.download_button(
            label="⬇️ Download Processed Video",
            data=open(processed_path, "rb").read(),
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

        # Return home button (reloads page)
        if st.button("🏠 Return Home"):
            st.experimental_rerun()
