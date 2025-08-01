# dashboard.py
import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
from datetime import datetime

# Set up paths
output_dir = '/Users/shubhadeepdas/Documents/data_science/projects/genai_transcript/output'
repo_path = os.path.join(output_dir, 'feature_repository')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(repo_path, exist_ok=True)

# Initialize Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Set it using 'export GOOGLE_API_KEY=your-key'.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# Placeholder functions
def generate_modus_operandi(transcript_text, transcript_id):
    """Generate MO from transcript text."""
    st.write("Placeholder: Generating MO...")
    return {"transcript_id": transcript_id, "modus_operandi": "Sample MO: Fraudster used stolen credentials."}

def generate_features(modus_operandi, transcript_id):
    """Generate recommended features based on MO."""
    st.write("Placeholder: Generating features...")
    return pd.DataFrame([
        {
            'transcript_id': transcript_id,
            'generated_modus_operandi': modus_operandi,
            'new_feature_name': 'sample_feature',
            'description': 'Sample feature description',
            'required_raw_variables': 'transaction_id,transaction_amt',
            'remark': 'Addresses missed fraud'
        }
    ])

def simulate_importance_scores(features_df):
    """Simulate importance scores for existing and recommended features."""
    st.write("Placeholder: Simulating importance scores...")
    return features_df.assign(importance_score=0.75)

def generate_shap_plots(features_df, transcript_id):
    """Generate SHAP plots for existing and recommended features."""
    st.write("Placeholder: Generating SHAP plots...")
    return {"before": "placeholder_plot_before.png", "after": "placeholder_plot_after.png"}

def save_to_feature_repository(approved_features, transcript_id):
    """Save approved features to feature repository."""
    st.write("Placeholder: Saving to feature repository...")
    # Placeholder: Append to features_history.parquet/csv and update features_latest.parquet/csv

# Streamlit layout
st.set_page_config(page_title="Feature Recommendation From Call Transcript Using GenAI POC", layout="wide")
st.title("Feature Recommendation Dashboard")

# Sidebar for file upload and history
with st.sidebar:
    st.header("Upload Transcript")
    uploaded_file = st.file_uploader("Upload a transcript text file", type=["txt"])
    
    st.header("Processed Transcripts")
    history_path = os.path.join(output_dir, 'transcript_history.csv')
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        st.dataframe(history_df[['transcript_id', 'modus_operandi']], use_container_width=True)
    else:
        st.write("No transcripts processed yet.")

# Main content
if uploaded_file:
    # Read transcript
    transcript_id = uploaded_file.name.replace('.txt', '')
    transcript_text = uploaded_file.read().decode('utf-8')
    
    st.header(f"Analysis for Transcript: {transcript_id}")
    
    # Section 1: Generate Modus Operandi
    st.subheader("Modus Operandi")
    mo_result = generate_modus_operandi(transcript_text, transcript_id)
    st.write(f"MO: {mo_result['modus_operandi']}")
    
    # Save MO to history
    history_entry = pd.DataFrame([{
        'transcript_id': transcript_id,
        'modus_operandi': mo_result['modus_operandi'],
        'processed_timestamp': datetime.now()
    }])
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        history_df = pd.concat([history_df, history_entry], ignore_index=True)
    else:
        history_df = history_entry
    history_df.to_csv(history_path, index=False)

    # Section 2: Generate Recommended Features
    st.subheader("Recommended Features")
    features_df = generate_features(mo_result['modus_operandi'], transcript_id)
    st.dataframe(features_df[['new_feature_name', 'description', 'required_raw_variables', 'remark']], use_container_width=True)

    # Section 3: Simulate Importance Scores
    st.subheader("Feature Importance Scores")
    features_with_scores = simulate_importance_scores(features_df)
    st.dataframe(features_with_scores[['new_feature_name', 'importance_score']], use_container_width=True)

    # Section 4: SHAP Plots
    st.subheader("SHAP Plots (Before vs. After)")
    shap_plots = generate_shap_plots(features_with_scores, transcript_id)
    col1, col2 = st.columns(2)
    with col1:
        st.write("Before Adding Features")
        st.image(shap_plots['before'], caption="SHAP Plot (Existing Features)")
    with col2:
        st.write("After Adding Recommended Features")
        st.image(shap_plots['after'], caption="SHAP Plot (With Recommended Features)")

    # Section 5: Approve Features
    st.subheader("Approve Features")
    selected_features = st.multiselect("Select features to approve:", features_df['new_feature_name'].tolist())
    if st.button("Approve Selected Features"):
        if selected_features:
            approved_features = features_with_scores[features_with_scores['new_feature_name'].isin(selected_features)]
            save_to_feature_repository(approved_features, transcript_id)
            st.success("Approved features saved to feature repository!")
        else:
            st.warning("Please select at least one feature to approve.")

    # Save outputs
    full_output_path = os.path.join(output_dir, f'{transcript_id}_full.csv')
    summary_output_path = os.path.join(output_dir, f'{transcript_id}.csv')
    features_with_scores.to_csv(full_output_path, index=False)
    approved_features = features_with_scores[features_with_scores['new_feature_name'].isin(selected_features)][['transcript_id', 'generated_modus_operandi', 'new_feature_name', 'description', 'importance_score']]
    approved_features.to_csv(summary_output_path, index=False)

else:
    st.info("Please upload a transcript file to begin analysis.")