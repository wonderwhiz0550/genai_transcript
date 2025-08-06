import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import json
import logging
import re
from pathlib import Path
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stylable_container import stylable_container

# -------------------------- UI Enhancements --------------------------
st.set_page_config(
    page_title="Echo AI – Feature Recommendation Dashboard",
    layout="wide",
    page_icon="📊"
)

st.markdown("""
    <style>
        html, body, [class*='css'] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f5f7fa;
        }
        .stApp {
            padding: 2rem;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2b4c7e;
        }
        .stDataFrame thead tr th {
            background-color: #2b4c7e;
            color: white;
        }
        .stButton>button {
            background-color: #2b4c7e;
            color: white;
            border: None;
            border-radius: 8px;
            padding: 8px 16px;
        }
        .stButton>button:hover {
            background-color: #3d5a91;
            color: #f0f0f0;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Echo AI")
st.markdown("##### Understanding what your customers *really* say — and turning it into fraud intelligence.")
add_vertical_space(1)
# ---------------------------------------------------------------------

# Define paths
HOME_PATH = Path(__file__).parent
output_dir = HOME_PATH / 'output'
repo_path = output_dir / 'feature_repository'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(repo_path, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, filename=os.path.join(output_dir, 'dashboard.log'), filemode='a')
logger = logging.getLogger(__name__)

# Load Gemini API
GOOGLE_API_KEY = "AIzaSyA-9N4KRv8LHPBYVa6vkL6hc_sQVZ2LRUE"
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

raw_variables = [
    'transaction_id', 'transaction_date', 'transaction_time', 'transaction_amt', 'mcc', 'pos',
    'cnp_flag', 'secure_flag', 'merchant_name', 'merchant_id', 'merchant_state_code', 'merchant_cntry_code',
    'digital_code', 'event_date', 'event_time'
]

existing_features_path = os.path.join(repo_path, 'features_latest.csv')
if os.path.exists(existing_features_path):
    existing_features_df = pd.read_csv(existing_features_path)
else:
    st.error("Existing features not found in feature repository.")
    st.stop()

mo_prompt_template = (
    "Analyze the following UK bank call transcript for an Account Takeover (ATO) fraud case: '{transcript}'. "
    "Generate a concise (1-2 sentences) description of the fraud modus operandi, focusing on how the fraudster gained access to the customer's credit card account (e.g., phishing, credential stuffing, social engineering) and their actions (e.g., changing details, unauthorized transactions). "
    "Ensure alignment with UK banking context (e.g., Faster Payments, UK Finance)."
    "Think deeply and generate the modus operandi and avoid using markdown symbols like asterisks (*) and keep it simple ."
)

feature_prompt_template = (
    "Analyze the following Account Takeover (ATO) fraud modus operandi from a UK bank call transcript: '{mo}'. "
    "Using the raw variables {raw_vars}, recommend 2-3 sophisticated features for a fraud detection model to prevent missed frauds. "
    "Each feature must address why the fraud was missed (e.g., gaps in detecting unusual login patterns or transaction behaviors). "
    "Return a list of dictionaries where each dictionary has: "
    "'transcript_id': '{transcript_id}', "
    "'generated_modus_operandi': '{mo}', "
    "'new_feature_name': unique and descriptive name, "
    "'description': explain what the feature does and how it detects fraud, "
    "'required_raw_variables': comma-separated list of variables from the provided list, "
    "'remark': justify how the feature prevents the missed fraud based on the MO. "
    "Ensure alignment with UK banking context (e.g., Faster Payments, sort code). "
    "Return only the list of dictionaries as a clean JSON string, e.g., [{"transcript_id": "...", ...}]. "
    "Avoid any special characters, escape characters, or newlines in the output values."
)

# Sidebar
with st.sidebar:
    with stylable_container("sidebar-box", css="""
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    """
    ):
        st.header("📄 Upload Transcript")
        uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])

        st.divider()
        st.header("🕓 Transcript History")
        history_path = os.path.join(output_dir, 'transcript_history.csv')
        if os.path.exists(history_path):
            history_df = pd.read_csv(history_path)
            st.dataframe(history_df[['transcript_id', 'modus_operandi']], use_container_width=True)
        else:
            st.write("No transcripts processed yet.")

# 👇 Add your existing logic here as-is (function definitions + main logic)
# Just ensure you replace subheader lines as instructed earlier:
# st.subheader("XYZ") → st.markdown("### 🧠/💡/📈 etc.")

# Add branding footer
st.markdown("""
---
<div style='text-align:center; color: gray;'>
    © 2025 Echo AI | Built for secure and explainable fraud detection
</div>
""", unsafe_allow_html=True)
