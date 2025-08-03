# dashboard.py
# Purpose: A Streamlit dashboard for analyzing UK bank call transcripts to detect Account Takeover (ATO) fraud.
#          Generates fraud modus operandi (MO), recommends detection features, simulates importance scores,
#          visualizes SHAP plots, and allows users to approve features for storage in a repository.
# Usage: Set HOME_PATH to the desired directory for outputs and feature repository, then run with `streamlit run dashboard.py`.

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

# Define home path for output and repository (modify this for your system)
HOME_PATH = '/Users/shubhadeepdas/Documents/data_science/projects/genai_transcript'  # Change this to your base directory
output_dir = os.path.join(HOME_PATH, 'output')
repo_path = os.path.join(output_dir, 'feature_repository')

# Configure logging to track script execution and errors
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(output_dir, 'dashboard.log'),
    filemode='a'
)
logger = logging.getLogger(__name__)

# Create output and repository directories if they don’t exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(repo_path, exist_ok=True)

# Initialize Gemini API for generating MO and features
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Set it using 'export GOOGLE_API_KEY=your-key'.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# Define raw variables available for feature generation
raw_variables = [
    'transaction_id', 'transaction_date', 'transaction_time', 'transaction_amt', 'mcc', 'pos',
    'cnp_flag', 'secure_flag', 'merchant_name', 'merchant_id', 'merchant_state_code', 'merchant_cntry_code',
    'digital_code', 'event_date', 'event_time'
]

# Load existing features from repository
existing_features_path = os.path.join(repo_path, 'features_latest.csv')
if os.path.exists(existing_features_path):
    existing_features_df = pd.read_csv(existing_features_path)
else:
    st.error("Existing features not found in feature repository.")
    st.stop()

# Define prompt template for generating fraud modus operandi (MO)
mo_prompt_template = (
    "Analyze the following UK bank call transcript for an Account Takeover (ATO) fraud case: '{transcript}'. "
    "Generate a concise (1-2 sentences) description of the fraud modus operandi, focusing on how the fraudster gained access to the customer's credit card account (e.g., phishing, credential stuffing, social engineering) and their actions (e.g., changing details, unauthorized transactions). "
    "Ensure alignment with UK banking context (e.g., Faster Payments, UK Finance)."
    "Think deeply and generate the modus operandi and avoid using markdown symbols like asterisks (*) and keep it simple ."
)

# Define prompt template for generating recommended features
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
    "Return only the list of dictionaries as a clean JSON string, e.g., [{{\"transcript_id\": \"...\", ...}}]. "
    "Avoid any special characters, escape characters, or newlines in the output values."
)

# Function: Generates a fraud modus operandi (MO) from a transcript using the Gemini API
# Input: transcript_text (str), transcript_id (str)
# Output: Dictionary with transcript_id and modus_operandi
def generate_modus_operandi(transcript_text, transcript_id):
    mo_prompt = mo_prompt_template.format(transcript=transcript_text)
    try:
        response = model.generate_content(mo_prompt, generation_config={'max_output_tokens': 5000, 'temperature': 0.7})
        generated_mo = response.text.strip()
        logger.info(f"MO generated for {transcript_id}: {generated_mo}")
        return {"transcript_id": transcript_id, "modus_operandi": generated_mo}
    except Exception as e:
        logger.error(f"Error generating MO for {transcript_id}: {e}")
        st.error(f"Error generating MO: {e}")
        return {"transcript_id": transcript_id, "modus_operandi": "Error generating MO"}

# Initialize Streamlit session state to persist data across reruns
# Purpose: Stores MO, features, scores, plots, and form state to prevent redundant computations
if 'features_df' not in st.session_state:
    st.session_state.features_df = pd.DataFrame([])
if 'features_with_scores' not in st.session_state:
    st.session_state.features_with_scores = pd.DataFrame([])
if 'mo_result' not in st.session_state:
    st.session_state.mo_result = {}
if 'shap_plots' not in st.session_state:
    st.session_state.shap_plots = {}
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = {}
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = {}

# Function: Generates recommended fraud detection features based on MO
# Input: modus_operandi (str), transcript_id (str)
# Output: DataFrame with recommended features or empty DataFrame on error
def generate_features(modus_operandi, transcript_id):
    # Format prompt and query Gemini API
    feature_prompt = feature_prompt_template.format(mo=modus_operandi, transcript_id=transcript_id, raw_vars=raw_variables)
    logger.info(f"Feature prompt for {transcript_id}: {feature_prompt[:500]}...")
    try:
        response = model.generate_content(feature_prompt, generation_config={'max_output_tokens': 30000, 'temperature': 0.7})
        feature_text = response.text.strip()
        logger.info(f"Raw feature response for {transcript_id}: {feature_text}")
        
        if not feature_text:
            logger.error(f"Empty feature response for {transcript_id}")
            return pd.DataFrame([])
        
        # Clean JSON response by removing code block markers
        feature_text = re.sub(r'^```json\s*|\s*```$', '', feature_text, flags=re.MULTILINE)
        feature_text = feature_text.strip()
        logger.info(f"Cleaned feature response for {transcript_id}: {feature_text}")
        
        # Parse JSON into DataFrame
        features = json.loads(feature_text)
        features_df = pd.DataFrame(features)
        
        # Verify required columns
        required_columns = ['transcript_id', 'generated_modus_operandi', 'new_feature_name', 'description', 'required_raw_variables', 'remark']
        if not all(col in features_df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in features_df.columns]
            logger.error(f"Missing columns in features_df for {transcript_id}: {missing_cols}")
            return pd.DataFrame([])
        
        # Clean and validate required_raw_variables
        def clean_raw_variables(raw_vars):
            if not isinstance(raw_vars, str):
                logger.warning(f"Invalid raw_vars type for {transcript_id}: {type(raw_vars)}")
                return ''
            # Split and clean the string
            vars_list = [var.strip() for var in raw_vars.split(',') if var.strip()]
            # Validate against raw_variables
            valid_vars = [var for var in vars_list if var in raw_variables]
            if len(valid_vars) != len(vars_list):
                logger.warning(f"Invalid variables in required_raw_variables for {transcript_id}: {vars_list}")
            return ','.join(valid_vars) if valid_vars else ''
        
        # Apply cleaning steps
        features_df = features_df.apply(lambda x: x.str.replace(r"\'", "'", regex=True) if x.dtype == "object" else x)
        features_df = features_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        features_df = features_df.fillna('')
        features_df['required_raw_variables'] = features_df['required_raw_variables'].apply(clean_raw_variables)
        logger.info(f"Processed features for {transcript_id}: {features_df.to_dict()}")
        return features_df
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing features JSON for {transcript_id}: {e}")
        return pd.DataFrame([])
    except Exception as e:
        logger.error(f"Error generating features for {transcript_id}: {e}")
        return pd.DataFrame([])

# Function: Simulates importance scores for recommended features
# Input: features_df (DataFrame)
# Output: DataFrame with added importance_score column
def simulate_importance_scores(features_df):
    if features_df.empty:
        return features_df
    
    # Load existing feature scores
    existing_scores = existing_features_df[['feature_name', 'importance_score']].set_index('feature_name').to_dict()['importance_score']
    
    # Simulate scores (0.5-0.9) based on required_raw_variables
    new_scores = []
    for _, row in features_df.iterrows():
        vars_list = row['required_raw_variables'].split(',') if row['required_raw_variables'] else []
        vars_list = [var.strip() for var in vars_list if var.strip()]
        score = 0.6  # Base score
        if any(var in ['digital_code', 'secure_flag', 'event_date', 'event_time'] for var in vars_list):
            score += 0.2  # Boost for login-related variables
        if any(var in ['transaction_amt', 'mcc', 'cnp_flag'] for var in vars_list):
            score += 0.1  # Boost for transaction-related variables
        score = min(0.9, score)  # Cap at 0.9
        new_scores.append(score)
    
    features_df = features_df.copy()  # Avoid modifying original
    features_df['importance_score'] = new_scores
    features_df['required_raw_variables'] = features_df['required_raw_variables'].apply(lambda x: x if isinstance(x, str) else '')
    logger.info(f"Simulated scores for features: {features_df[['new_feature_name', 'required_raw_variables', 'importance_score']].to_dict()}")
    return features_df

# Function: Generates SHAP plots to compare model performance before/after adding recommended features
# Input: features_df (DataFrame), transcript_id (str)
# Output: Dictionary with paths to before/after SHAP plot images
def generate_shap_plots(features_df, transcript_id):
    # Generate synthetic data for SHAP analysis
    np.random.seed(42)
    n_samples = 100
    feature_names = existing_features_df['feature_name'].tolist()
    data = {
        'cust_day_since_last_failed_login': np.random.exponential(10, n_samples),  # Days, exponential
        'card_cnt_tran_mcc_7d': np.random.poisson(5, n_samples),  # Count, Poisson
        'cust_cnt_tran_pos_30d': np.random.poisson(10, n_samples),  # Count, Poisson
        'card_ratio_mcc_1d': np.random.uniform(0, 1, n_samples),  # Ratio, uniform
        'card_cnt_tran_night_90d': np.random.poisson(3, n_samples),  # Count, Poisson
        'cust_day_since_last_mid': np.random.exponential(15, n_samples),  # Days, exponential
        'card_cnt_tran_secure_12h': np.random.poisson(2, n_samples),  # Count, Poisson
        'card_normed_amt_30d': np.random.normal(0, 1, n_samples)  # Normalized, normal
    }
    X = pd.DataFrame(data)
    y = np.random.choice([0, 1], n_samples)  # Binary fraud labels
    
    # Train logistic regression model on existing features
    model_lr = LogisticRegression().fit(X, y)
    
    # Generate SHAP values for existing features
    explainer = shap.LinearExplainer(model_lr, X)
    shap_values = explainer.shap_values(X)
    
    # Plot SHAP summary (before)
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    before_plot = os.path.join(output_dir, f'{transcript_id}_shap_before.png')
    plt.savefig(before_plot, bbox_inches='tight')
    plt.close()
    
    # Add recommended features to synthetic data
    X_extended = X.copy()
    for _, row in features_df.iterrows():
        feature_name = row['new_feature_name']
        vars_list = row['required_raw_variables'].split(',')
        if any(var in ['digital_code', 'event_date', 'event_time'] for var in vars_list):
            X_extended[feature_name] = np.random.exponential(10, n_samples)  # Login-like
        elif 'transaction_amt' in vars_list:
            X_extended[feature_name] = np.random.normal(0, 1, n_samples)  # Amount-like
        else:
            X_extended[feature_name] = np.random.poisson(5, n_samples)  # Count-like
    
    # Train model with extended features
    model_lr_extended = LogisticRegression().fit(X_extended, y)
    explainer_extended = shap.LinearExplainer(model_lr_extended, X_extended)
    shap_values_extended = explainer_extended.shap_values(X_extended)
    
    # Plot SHAP summary (after)
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values_extended, X_extended, feature_names=X_extended.columns, show=False)
    after_plot = os.path.join(output_dir, f'{transcript_id}_shap_after.png')
    plt.savefig(after_plot, bbox_inches='tight')
    plt.close()
    
    return {"before": before_plot, "after": after_plot}

# Function: Saves approved features to the feature repository
# Input: approved_features (DataFrame), transcript_id (str)
# Output: None (saves to features_history.csv/parquet and features_latest.csv/parquet)
def save_to_feature_repository(approved_features, transcript_id):
    history_path = os.path.join(repo_path, 'features_history.csv')
    latest_path = os.path.join(repo_path, 'features_latest.csv')
    
    # Prepare approved features
    approved_features = approved_features.copy()
    # Reload existing features to ensure unique feature IDs
    global existing_features_df
    if os.path.exists(latest_path):
        existing_features_df = pd.read_csv(latest_path)
    else:
        existing_features_df = pd.DataFrame(columns=['feature_id', 'feature_name', 'description', 'required_raw_variables', 'importance_score', 'created_timestamp'])
    
    approved_features['feature_id'] = [f'feat_{len(existing_features_df) + i + 1}' for i in range(len(approved_features))]
    approved_features['created_timestamp'] = datetime.now().isoformat()
    approved_features = approved_features.rename(columns={'new_feature_name': 'feature_name'})
    
    # Append to feature history
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        history_df = pd.concat([history_df, approved_features[['feature_id', 'feature_name', 'description', 'required_raw_variables', 'importance_score', 'created_timestamp']]], ignore_index=True)
    else:
        history_df = approved_features[['feature_id', 'feature_name', 'description', 'required_raw_variables', 'importance_score', 'created_timestamp']]
    history_df.to_csv(history_path, index=False)
    history_df.to_parquet(os.path.join(repo_path, 'features_history.parquet'), index=False)
    
    # Update latest features
    existing_features_df = pd.concat([existing_features_df, approved_features[['feature_id', 'feature_name', 'description', 'required_raw_variables', 'importance_score', 'created_timestamp']]], ignore_index=True)
    existing_features_df.to_csv(latest_path, index=False)
    existing_features_df.to_parquet(os.path.join(repo_path, 'features_latest.parquet'), index=False)
    logger.info(f"Saved approved features for {transcript_id} to repository: {approved_features[['feature_name']].to_dict()}")

# Configure Streamlit page layout
# Purpose: Sets up a wide layout for the dashboard with a title
st.set_page_config(page_title="Feature Recommendation POC", layout="wide")
st.title("AI Powered Feature Recommendation From Call Transcript")

# Sidebar: Handles file upload and displays transcript history
# Purpose: Allows users to upload .txt transcripts and view processed transcripts
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

# Main Content: Processes uploaded transcript through analysis steps
# Purpose: Displays MO, features, scores, SHAP plots, and allows feature approval
if uploaded_file:
    # Read transcript and extract ID
    transcript_id = uploaded_file.name.replace('.txt', '')
    transcript_text = uploaded_file.read().decode('utf-8')
    
    st.header(f"Analysis for Transcript: {transcript_id}")
    
    # Section 1: Generate Modus Operandi
    # Purpose: Generates and displays the fraud MO using the Gemini API
    st.subheader("Modus Operandi:")
    if not st.session_state.mo_result.get('transcript_id') == transcript_id:
        mo_result = generate_modus_operandi(transcript_text, transcript_id)
        st.session_state.mo_result = mo_result
    else:
        mo_result = st.session_state.mo_result
    st.write(f"Key Modus Operandi: {mo_result['modus_operandi']}")
    
    # Save MO to history
    history_entry = pd.DataFrame([{
        'transcript_id': transcript_id,
        'modus_operandi': mo_result['modus_operandi'],
        'processed_timestamp': datetime.now().isoformat()
    }])
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        history_df = pd.concat([history_df, history_entry], ignore_index=True)
    else:
        history_df = history_entry
    history_df.to_csv(history_path, index=False)
    
    # Save MO to full output
    full_output = {'transcript_id': transcript_id, 'modus_operandi': mo_result['modus_operandi']}

    # Section 2: Generate Recommended Features
    # Purpose: Generates and displays recommended fraud detection features
    st.subheader("Recommended Features:")
    if not st.session_state.features_df.empty and st.session_state.features_df.get('transcript_id', [''])[0] == transcript_id:
        features_df = st.session_state.features_df
    else:
        features_df = generate_features(mo_result['modus_operandi'], transcript_id)
        st.session_state.features_df = features_df
    if not features_df.empty and 'new_feature_name' in features_df.columns:
        st.dataframe(features_df[['new_feature_name', 'description', 'required_raw_variables', 'remark']], use_container_width=True)
    else:
        st.error("No features generated or invalid feature data. Check dashboard.log for details.")
        st.stop()
    
    # Section 3: Simulate Importance Scores
    # Purpose: Assigns simulated importance scores to recommended features
    st.subheader("Feature Importance Scores")
    if not st.session_state.features_with_scores.empty and st.session_state.features_with_scores.get('transcript_id', [''])[0] == transcript_id:
        features_with_scores = st.session_state.features_with_scores
    else:
        features_with_scores = simulate_importance_scores(features_df)
        st.session_state.features_with_scores = features_with_scores
    if not features_with_scores.empty:
        st.dataframe(features_with_scores[['new_feature_name', 'importance_score']], use_container_width=True)
    
    # Display all features (existing + recommended)
    all_features_scores = pd.concat([
        existing_features_df[['feature_name', 'importance_score']].rename(columns={'feature_name': 'new_feature_name'}),
        features_with_scores[['new_feature_name', 'importance_score']]
    ], ignore_index=True)
    st.write("All Features (Existing + Recommended):")
    st.dataframe(all_features_scores, use_container_width=True)

    # Section 4: SHAP Plots
    # Purpose: Visualizes SHAP plots to compare model performance before/after adding features
    st.subheader("SHAP Plots (Before vs. After)")
    if st.session_state.shap_plots.get('transcript_id') == transcript_id:
        shap_plots = st.session_state.shap_plots
    else:
        shap_plots = generate_shap_plots(features_with_scores, transcript_id)
        st.session_state.shap_plots = shap_plots
        st.session_state.shap_plots['transcript_id'] = transcript_id
    col1, col2 = st.columns(2)
    with col1:
        st.write("Before Adding Features")
        st.image(shap_plots['before'], caption="SHAP Plot (Existing Features)")
    with col2:
        st.write("After Adding Recommended Features")
        st.image(shap_plots['after'], caption="SHAP Plot (With Recommended Features)")

    # Save full output to CSV
    full_output_df = features_with_scores.copy()
    full_output_df['modus_operandi'] = mo_result['modus_operandi']
    full_output_path = os.path.join(output_dir, f'{transcript_id}_full.csv')
    full_output_df.to_csv(full_output_path, index=False)

    # Section 5: Approve Features
    # Purpose: Allows users to select and approve features, saving them to the repository
    def handle_form_submission(transcript_id):
        selected_features = st.session_state[f'multiselect_{transcript_id}']
        logger.info(f"Form submitted for {transcript_id}, selected features: {selected_features}")
        if selected_features:
            approved_features = st.session_state.features_df[st.session_state.features_df['new_feature_name'].isin(selected_features)]
            approved_features_with_scores = simulate_importance_scores(approved_features)
            save_to_feature_repository(approved_features_with_scores, transcript_id)
            st.success("Approved features saved to feature repository!")
            
            # Save summary output
            summary_output = approved_features_with_scores[['transcript_id', 'generated_modus_operandi', 'new_feature_name', 'description', 'importance_score']]
            summary_output = summary_output.rename(columns={'generated_modus_operandi': 'modus_operandi'})
            summary_output_path = os.path.join(output_dir, f'{transcript_id}_summary.csv')
            summary_output.to_csv(summary_output_path, index=False)
            st.write("Summary saved to:", summary_output_path)
            
            # Clear selections and mark form as submitted
            st.session_state.selected_features[transcript_id] = []
            st.session_state.form_submitted[transcript_id] = True
        else:
            st.warning("Please select at least one feature to approve.")
            logger.info(f"No features selected for approval in {transcript_id}")

    st.subheader("Approve Features")
    if not st.session_state.features_df.empty and 'new_feature_name' in st.session_state.features_df.columns:
        with st.form(key=f"approve_features_form_{transcript_id}"):
            selected_features = st.multiselect(
                "Select features to approve:",
                options=st.session_state.features_df['new_feature_name'].tolist(),
                default=st.session_state.selected_features.get(transcript_id, []),
                key=f"multiselect_{transcript_id}"
            )
            st.form_submit_button("Approve Selected Features", on_click=handle_form_submission, args=(transcript_id,))
    else:
        st.error("Cannot approve features: No valid features available. Check dashboard.log for details.")
else:
    st.info("Please upload a transcript file to begin analysis.")