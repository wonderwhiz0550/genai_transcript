# dashboard_styled.py
# Purpose: A Streamlit dashboard for analyzing UK bank call transcripts to detect Account Takeover (ATO) fraud.
#          Generates fraud modus operandi (MO), recommends detection features, simulates importance scores,
#          visualizes SHAP plots, and allows users to approve features for storage in a repository.

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

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="GenAI Transcript Intelligence",
    page_icon="📊",
    layout="wide",
)

# -------------------------------------------------------------------
# GLOBAL DARK THEME STYLING
# -------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Global background + font */
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #0f172a 0, #020617 45%, #020617 100%) !important;
    color: #e5e7eb !important;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #020617 !important;
    border-right: 1px solid rgba(148, 163, 184, 0.3);
}
section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* Block container padding */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1.5rem;
}

/* Headings */
h1, h2, h3 {
    font-weight: 600 !important;
}

/* Metric cards */
.metric-card {
    padding: 1.1rem 1.3rem;
    border-radius: 1rem;
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.25);
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.6);
}

/* Section cards */
.section-card {
    padding: 1.2rem 1.4rem;
    border-radius: 1.2rem;
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid rgba(148, 163, 184, 0.3);
    box-shadow: 0 20px 50px rgba(15, 23, 42, 0.75);
    margin-bottom: 1.5rem;
}

/* Buttons */
button[kind="primary"],
div.stButton > button {
    border-radius: 999px !important;
    border: 1px solid rgba(248, 250, 252, 0.12) !important;
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #f9fafb !important;
    font-weight: 500 !important;
    padding: 0.45rem 1.4rem !important;
}
div.stButton > button:hover {
    filter: brightness(1.06);
    box-shadow: 0 10px 30px rgba(88, 80, 236, 0.55);
}

/* Textareas + inputs */
textarea, input, select {
    border-radius: 0.75rem !important;
    border: 1px solid rgba(148, 163, 184, 0.5) !important;
}
textarea:focus, input:focus, select:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 1px #6366f1 !important;
}

/* Dataframes / tables */
[data-testid="stDataFrame"] {
    border-radius: 0.9rem;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.35);
    background: rgba(15, 23, 42, 0.9);
}

/* Hide top-right Streamlit menu + footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background: transparent;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------------
# PATHS, LOGGING, STORAGE SETUP
# -------------------------------------------------------------------
HOME_PATH = Path(__file__).resolve().parent
output_dir = HOME_PATH / "output"
repo_path = output_dir / "feature_repository"

output_dir.mkdir(parents=True, exist_ok=True)
repo_path.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=str(output_dir / "dashboard.log"),
    filemode="a",
)
logger = logging.getLogger(__name__)

# Paths for existing features + transcript history
existing_features_path = repo_path / "features_latest.csv"
history_path = output_dir / "transcript_history.csv"

# -------------------------------------------------------------------
# LOAD EXISTING FEATURES (GRACEFUL IF MISSING)
# -------------------------------------------------------------------
if existing_features_path.exists():
    existing_features_df = pd.read_csv(existing_features_path)
else:
    # Start with an empty feature repository
    existing_features_df = pd.DataFrame(
        columns=[
            "feature_id",
            "feature_name",
            "description",
            "required_raw_variables",
            "importance_score",
            "created_timestamp",
        ]
    )
    st.warning(
        "Existing features not found in feature repository. "
        "Starting with an empty repository."
    )

# -------------------------------------------------------------------
# LOAD TRANSCRIPT HISTORY (FOR KPI + SIDEBAR)
# -------------------------------------------------------------------
if history_path.exists():
    transcript_history_df = pd.read_csv(history_path)
else:
    transcript_history_df = pd.DataFrame(
        columns=["transcript_id", "modus_operandi", "processed_timestamp"]
    )

# -------------------------------------------------------------------
# INITIAL KPI VALUES
# -------------------------------------------------------------------
total_transcripts = len(transcript_history_df) if not transcript_history_df.empty else 0
total_existing_features = (
    len(existing_features_df["feature_name"].unique())
    if not existing_features_df.empty
    else 0
)

# -------------------------------------------------------------------
# GEMINI API CONFIG
# -------------------------------------------------------------------
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error(
        "GOOGLE_API_KEY not found.\n\n"
        "Go to your app → Settings → Secrets and add:\n\n"
        'GOOGLE_API_KEY = "your-gemini-api-key-here"'
    )
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# -------------------------------------------------------------------
# DOMAIN CONSTANTS & PROMPTS
# -------------------------------------------------------------------
raw_variables = [
    "transaction_id",
    "transaction_date",
    "transaction_time",
    "transaction_amt",
    "mcc",
    "pos",
    "cnp_flag",
    "secure_flag",
    "merchant_name",
    "merchant_id",
    "merchant_state_code",
    "merchant_cntry_code",
    "digital_code",
    "event_date",
    "event_time",
]

mo_prompt_template = (
    "Analyze the following UK bank call transcript for an Account Takeover (ATO) fraud case: '{transcript}'. "
    "Generate a concise (1-2 sentences) description of the fraud modus operandi, focusing on how the fraudster "
    "gained access to the customer's credit card account (e.g., phishing, credential stuffing, social engineering) "
    "and their actions (e.g., changing details, unauthorized transactions). "
    "Ensure alignment with UK banking context (e.g., Faster Payments, UK Finance). "
    "Think deeply and generate the modus operandi and avoid using markdown symbols like asterisks (*) and keep it simple."
)

feature_prompt_template = (
    "Analyze the following Account Takeover (ATO) fraud modus operandi from a UK bank call transcript: '{mo}'. "
    "Using the raw variables {raw_vars}, recommend 2-3 sophisticated features for a fraud detection model "
    "to prevent missed frauds. Each feature must address why the fraud was missed "
    "(e.g., gaps in detecting unusual login patterns or transaction behaviors). "
    "Return a list of dictionaries where each dictionary has: "
    "'transcript_id': '{transcript_id}', "
    "'generated_modus_operandi': '{mo}', "
    "'new_feature_name': unique and descriptive name, "
    "'description': explain what the feature does and how it detects fraud, "
    "'required_raw_variables': comma-separated list of variables from the provided list, "
    "'remark': justify how the feature prevents the missed fraud based on the MO. "
    "Ensure alignment with UK banking context (e.g., Faster Payments, sort code). "
    "Return only the list of dictionaries as a clean JSON string, e.g., "
    '[{{\"transcript_id\": \"...\", ...}}]. '
    "Avoid any special characters, escape characters, or newlines in the output values."
)

# -------------------------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------------------------
if "features_df" not in st.session_state:
    st.session_state.features_df = pd.DataFrame([])
if "features_with_scores" not in st.session_state:
    st.session_state.features_with_scores = pd.DataFrame([])
if "mo_result" not in st.session_state:
    st.session_state.mo_result = {}
if "shap_plots" not in st.session_state:
    st.session_state.shap_plots = {}
if "selected_features" not in st.session_state:
    st.session_state.selected_features = {}
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = {}

# -------------------------------------------------------------------
# HELPER BANNERS
# -------------------------------------------------------------------
def success_banner(text: str):
    st.markdown(
        f"""
        <div style="
            border-radius:0.9rem;
            border:1px solid rgba(34,197,94,0.4);
            background:rgba(22,163,74,0.18);
            padding:0.75rem 1rem;
            font-size:0.9rem;
            margin-bottom:0.75rem;
        ">
            ✅ {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def warning_banner(text: str):
    st.markdown(
        f"""
        <div style="
            border-radius:0.9rem;
            border:1px solid rgba(250,204,21,0.4);
            background:rgba(202,138,4,0.18);
            padding:0.75rem 1rem;
            font-size:0.9rem;
            margin-bottom:0.75rem;
        ">
            ⚠️ {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------
# CORE FUNCTIONS (MO, FEATURES, SHAP, SAVE)
# -------------------------------------------------------------------
def generate_modus_operandi(transcript_text: str, transcript_id: str) -> dict:
    """Generate a fraud modus operandi description using Gemini."""
    mo_prompt = mo_prompt_template.format(transcript=transcript_text)
    try:
        response = model.generate_content(
            mo_prompt,
            generation_config={"max_output_tokens": 5000, "temperature": 0.7},
        )
        generated_mo = response.text.strip()
        logger.info(f"MO generated for {transcript_id}: {generated_mo}")
        return {"transcript_id": transcript_id, "modus_operandi": generated_mo}
    except Exception as e:
        logger.error(f"Error generating MO for {transcript_id}: {e}")
        st.error(f"Error generating MO: {e}")
        return {"transcript_id": transcript_id, "modus_operandi": "Error generating MO"}


def generate_features(modus_operandi: str, transcript_id: str) -> pd.DataFrame:
    """Generate recommended fraud detection features based on MO."""
    feature_prompt = feature_prompt_template.format(
        mo=modus_operandi,
        transcript_id=transcript_id,
        raw_vars=raw_variables,
    )
    logger.info(f"Feature prompt for {transcript_id}: {feature_prompt[:500]}...")
    try:
        response = model.generate_content(
            feature_prompt,
            generation_config={"max_output_tokens": 30000, "temperature": 0.7},
        )
        feature_text = response.text.strip()
        logger.info(f"Raw feature response for {transcript_id}: {feature_text}")

        if not feature_text:
            logger.error(f"Empty feature response for {transcript_id}")
            return pd.DataFrame([])

        # Clean JSON response
        feature_text = re.sub(
            r"^```json\s*|\s*```$", "", feature_text, flags=re.MULTILINE
        ).strip()
        logger.info(f"Cleaned feature response for {transcript_id}: {feature_text}")

        # Parse JSON into DataFrame
        features = json.loads(feature_text)
        features_df = pd.DataFrame(features)

        required_columns = [
            "transcript_id",
            "generated_modus_operandi",
            "new_feature_name",
            "description",
            "required_raw_variables",
            "remark",
        ]
        if not all(col in features_df.columns for col in required_columns):
            missing_cols = [
                col for col in required_columns if col not in features_df.columns
            ]
            logger.error(
                f"Missing columns in features_df for {transcript_id}: {missing_cols}"
            )
            return pd.DataFrame([])

        def clean_raw_variables(raw_vars):
            if not isinstance(raw_vars, str):
                logger.warning(
                    f"Invalid raw_vars type for {transcript_id}: {type(raw_vars)}"
                )
                return ""
            vars_list = [var.strip() for var in raw_vars.split(",") if var.strip()]
            valid_vars = [var for var in vars_list if var in raw_variables]
            if len(valid_vars) != len(vars_list):
                logger.warning(
                    f"Invalid variables in required_raw_variables for "
                    f"{transcript_id}: {vars_list}"
                )
            return ",".join(valid_vars) if valid_vars else ""

        features_df = features_df.apply(
            lambda x: x.str.replace(r"\'", "'", regex=True)
            if x.dtype == "object"
            else x
        )
        features_df = features_df.apply(
            lambda x: x.str.strip() if x.dtype == "object" else x
        )
        features_df = features_df.fillna("")
        features_df["required_raw_variables"] = features_df[
            "required_raw_variables"
        ].apply(clean_raw_variables)

        logger.info(
            f"Processed features for {transcript_id}: {features_df.to_dict()}"
        )
        return features_df
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing features JSON for {transcript_id}: {e}")
        return pd.DataFrame([])
    except Exception as e:
        logger.error(f"Error generating features for {transcript_id}: {e}")
        return pd.DataFrame([])


def simulate_importance_scores(features_df: pd.DataFrame) -> pd.DataFrame:
    """Simulate importance scores (0.6–0.9) based on required_raw_variables."""
    if features_df.empty:
        return features_df

    # Existing scores (currently not used directly but kept for compatibility)
    _ = (
        existing_features_df[["feature_name", "importance_score"]]
        .set_index("feature_name")
        .to_dict()
        .get("importance_score", {})
        if not existing_features_df.empty
        else {}
    )

    new_scores = []
    for _, row in features_df.iterrows():
        vars_list = (
            row["required_raw_variables"].split(",")
            if row["required_raw_variables"]
            else []
        )
        vars_list = [var.strip() for var in vars_list if var.strip()]
        score = 0.6
        if any(
            var in ["digital_code", "secure_flag", "event_date", "event_time"]
            for var in vars_list
        ):
            score += 0.2
        if any(var in ["transaction_amt", "mcc", "cnp_flag"] for var in vars_list):
            score += 0.1
        score = min(0.9, score)
        new_scores.append(score)

    features_df = features_df.copy()
    features_df["importance_score"] = new_scores
    features_df["required_raw_variables"] = features_df["required_raw_variables"].apply(
        lambda x: x if isinstance(x, str) else ""
    )
    logger.info(
        "Simulated scores for features: "
        f"{features_df[['new_feature_name', 'required_raw_variables', 'importance_score']].to_dict()}"
    )
    return features_df


def generate_shap_plots(features_df: pd.DataFrame, transcript_id: str) -> dict:
    """Generate SHAP plots (before vs after) using synthetic data."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "cust_day_since_last_failed_login": np.random.exponential(10, n_samples),
        "card_cnt_tran_mcc_7d": np.random.poisson(5, n_samples),
        "cust_cnt_tran_pos_30d": np.random.poisson(10, n_samples),
        "card_ratio_mcc_1d": np.random.uniform(0, 1, n_samples),
        "card_cnt_tran_night_90d": np.random.poisson(3, n_samples),
        "cust_day_since_last_mid": np.random.exponential(15, n_samples),
        "card_cnt_tran_secure_12h": np.random.poisson(2, n_samples),
        "card_normed_amt_30d": np.random.normal(0, 1, n_samples),
    }
    X = pd.DataFrame(data)
    y = np.random.choice([0, 1], n_samples)

    feature_names = (
        existing_features_df["feature_name"].tolist()
        if not existing_features_df.empty
        else list(data.keys())
    )

    model_lr = LogisticRegression().fit(X, y)
    explainer = shap.LinearExplainer(model_lr, X)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    before_plot = output_dir / f"{transcript_id}_shap_before.png"
    plt.savefig(before_plot, bbox_inches="tight")
    plt.close()

    X_extended = X.copy()
    for _, row in features_df.iterrows():
        feature_name = row["new_feature_name"]
        vars_list = row["required_raw_variables"].split(",")
        if any(
            var in ["digital_code", "event_date", "event_time"] for var in vars_list
        ):
            X_extended[feature_name] = np.random.exponential(10, n_samples)
        elif "transaction_amt" in vars_list:
            X_extended[feature_name] = np.random.normal(0, 1, n_samples)
        else:
            X_extended[feature_name] = np.random.poisson(5, n_samples)

    model_lr_extended = LogisticRegression().fit(X_extended, y)
    explainer_extended = shap.LinearExplainer(model_lr_extended, X_extended)
    shap_values_extended = explainer_extended.shap_values(X_extended)

    plt.figure(figsize=(10, 5))
    shap.summary_plot(
        shap_values_extended, X_extended, feature_names=X_extended.columns, show=False
    )
    after_plot = output_dir / f"{transcript_id}_shap_after.png"
    plt.savefig(after_plot, bbox_inches="tight")
    plt.close()

    return {"before": str(before_plot), "after": str(after_plot)}


def save_to_feature_repository(approved_features: pd.DataFrame, transcript_id: str):
    """Persist approved features to repository (history + latest)."""
    history_repo_path = repo_path / "features_history.csv"
    latest_path = repo_path / "features_latest.csv"

    approved_features = approved_features.copy()

    global existing_features_df
    if latest_path.exists():
        existing_features_df = pd.read_csv(latest_path)
    else:
        existing_features_df = pd.DataFrame(
            columns=[
                "feature_id",
                "feature_name",
                "description",
                "required_raw_variables",
                "importance_score",
                "created_timestamp",
            ]
        )

    approved_features["feature_id"] = [
        f"feat_{len(existing_features_df) + i + 1}"
        for i in range(len(approved_features))
    ]
    approved_features["created_timestamp"] = datetime.now().isoformat()
    approved_features = approved_features.rename(
        columns={"new_feature_name": "feature_name"}
    )

    # History
    if history_repo_path.exists():
        history_df = pd.read_csv(history_repo_path)
        history_df = pd.concat(
            [
                history_df,
                approved_features[
                    [
                        "feature_id",
                        "feature_name",
                        "description",
                        "required_raw_variables",
                        "importance_score",
                        "created_timestamp",
                    ]
                ],
            ],
            ignore_index=True,
        )
    else:
        history_df = approved_features[
            [
                "feature_id",
                "feature_name",
                "description",
                "required_raw_variables",
                "importance_score",
                "created_timestamp",
            ]
        ]

    history_df.to_csv(history_repo_path, index=False)
    history_df.to_parquet(repo_path / "features_history.parquet", index=False)

    # Latest
    existing_features_df = pd.concat(
        [
            existing_features_df,
            approved_features[
                [
                    "feature_id",
                    "feature_name",
                    "description",
                    "required_raw_variables",
                    "importance_score",
                    "created_timestamp",
                ]
            ],
        ],
        ignore_index=True,
    )
    existing_features_df.to_csv(latest_path, index=False)
    existing_features_df.to_parquet(repo_path / "features_latest.parquet", index=False)

    logger.info(
        f"Saved approved features for {transcript_id} to repository: "
        f"{approved_features[['feature_name']].to_dict()}"
    )

# -------------------------------------------------------------------
# HEADER + KPI STRIP
# -------------------------------------------------------------------
st.markdown(
    """
    <div style="
        display:flex;
        align-items:center;
        justify-content:space-between;
        margin-bottom:1.5rem;
    ">
        <div>
            <h1 style="margin-bottom:0.3rem;">GenAI Transcript Intelligence</h1>
            <p style="margin-top:0; color:#9ca3af; font-size:0.95rem;">
                Generate synthetic fraud call transcripts, extract modus operandi,
                and design production-ready features for ATO detection.
            </p>
        </div>
        <div style="
            padding:0.6rem 1.1rem;
            border-radius:999px;
            border:1px solid rgba(148,163,184,0.35);
            background:rgba(15,23,42,0.95);
            font-size:0.8rem;
            color:#e5e7eb;
        ">
            🔐 Gemini powered · Internal risk analytics tool
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
with kpi_col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.caption("TOTAL TRANSCRIPTS PROCESSED")
    st.metric(label="", value=str(total_transcripts))
    st.markdown("</div>", unsafe_allow_html=True)

with kpi_col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.caption("FEATURES IN REPOSITORY")
    st.metric(label="", value=str(total_existing_features))
    st.markdown("</div>", unsafe_allow_html=True)

with kpi_col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.caption("LATEST RUN STATUS")
    st.metric(label="", value="Ready")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")  # small spacer

# -------------------------------------------------------------------
# SIDEBAR – CONFIG + UPLOAD + HISTORY
# -------------------------------------------------------------------
with st.sidebar:
    st.subheader("Scenario configuration")
    scenario_type = st.selectbox(
        "Scenario type",
        [
            "Card fraud",
            "UPI fraud (proxy)",
            "Loan scam",
            "Phishing / Vishing",
        ],
        index=0,
    )

    st.caption(
        "💡 Scenario type is only used as context for you while interpreting the MO and features."
    )

    st.markdown("---")

    st.subheader("Upload transcript")
    uploaded_file = st.file_uploader(
        "Upload a transcript text file",
        type=["txt"],
        help="Upload one .txt file containing the call transcript.",
    )

    st.markdown("---")
    st.subheader("Processed transcripts")
    if not transcript_history_df.empty:
        st.dataframe(
            transcript_history_df[["transcript_id", "modus_operandi"]],
            use_container_width=True,
            height=250,
        )
    else:
        st.write("No transcripts processed yet.")

# -------------------------------------------------------------------
# MAIN WORKFLOW
# -------------------------------------------------------------------
if uploaded_file:
    # --------------------- READ TRANSCRIPT --------------------------
    transcript_id = uploaded_file.name.replace(".txt", "")
    transcript_text = uploaded_file.read().decode("utf-8")

    # Generate / reuse MO
    if st.session_state.mo_result.get("transcript_id") != transcript_id:
        mo_result = generate_modus_operandi(transcript_text, transcript_id)
        st.session_state.mo_result = mo_result
    else:
        mo_result = st.session_state.mo_result

    # Update transcript history
    history_entry = pd.DataFrame(
        [
            {
                "transcript_id": transcript_id,
                "modus_operandi": mo_result["modus_operandi"],
                "processed_timestamp": datetime.now().isoformat(),
            }
        ]
    )
    if history_path.exists():
        existing_history = pd.read_csv(history_path)
        history_combined = pd.concat([existing_history, history_entry], ignore_index=True)
    else:
        history_combined = history_entry
    history_combined.to_csv(history_path, index=False)

    # Generate / reuse features
    if (
        not st.session_state.features_df.empty
        and "transcript_id" in st.session_state.features_df.columns
        and st.session_state.features_df["transcript_id"].iloc[0] == transcript_id
    ):
        features_df = st.session_state.features_df
    else:
        features_df = generate_features(mo_result["modus_operandi"], transcript_id)
        st.session_state.features_df = features_df

    # Guard: if no features, stop gracefully
    if features_df.empty or "new_feature_name" not in features_df.columns:
        warning_banner(
            "No features generated or invalid feature data. "
            "Please check the Gemini response and dashboard.log."
        )
        st.stop()

    # Generate / reuse importance scores
    if (
        not st.session_state.features_with_scores.empty
        and "transcript_id" in st.session_state.features_with_scores.columns
        and st.session_state.features_with_scores["transcript_id"].iloc[0]
        == transcript_id
    ):
        features_with_scores = st.session_state.features_with_scores
    else:
        features_with_scores = simulate_importance_scores(features_df)
        # Keep transcript_id in DF for session reuse check
        if "transcript_id" not in features_with_scores.columns:
            features_with_scores["transcript_id"] = transcript_id
        st.session_state.features_with_scores = features_with_scores

    # SHAP plots (reuse where possible)
    if st.session_state.shap_plots.get("transcript_id") == transcript_id:
        shap_plots = st.session_state.shap_plots
    else:
        shap_plots = generate_shap_plots(features_with_scores, transcript_id)
        shap_plots["transcript_id"] = transcript_id
        st.session_state.shap_plots = shap_plots

    # Save full output CSV for this run
    full_output_df = features_with_scores.copy()
    full_output_df["modus_operandi"] = mo_result["modus_operandi"]
    full_output_path = output_dir / f"{transcript_id}_full.csv"
    full_output_df.to_csv(full_output_path, index=False)

    # ----------------------------------------------------------------
    # TABS LAYOUT
    # ----------------------------------------------------------------
    tab_transcript, tab_mo, tab_features = st.tabs(
        ["① Transcript", "② Modus Operandi", "③ Feature Design & Impact"]
    )

    # ---------------------- TAB 1: TRANSCRIPT -----------------------
    with tab_transcript:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Transcript workspace")

        st.markdown(
            f"<p style='color:#9ca3af;font-size:0.9rem;'>"
            f"Scenario: <b>{scenario_type}</b> · Transcript ID: <code>{transcript_id}</code>"
            f"</p>",
            unsafe_allow_html=True,
        )

        st.text_area(
            "Transcript text",
            value=transcript_text,
            height=260,
            help="This is the raw call transcript used to derive modus operandi and feature suggestions.",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------- TAB 2: MODUS OPERANDI -------------------
    with tab_mo:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Modus operandi & risk narrative")

        st.markdown(
            "Gemini summarises how the fraud is executed, key red flags, and potential monitoring rules."
        )
        st.markdown("**Key modus operandi**")
        st.write(mo_result["modus_operandi"])

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------- TAB 3: FEATURES & IMPACT ----------------
    def handle_form_submission(transcript_id_inner: str):
        selected_features = st.session_state.get(f"multiselect_{transcript_id_inner}", [])
        logger.info(
            f"Form submitted for {transcript_id_inner}, selected features: {selected_features}"
        )
        if selected_features:
            approved_features = st.session_state.features_df[
                st.session_state.features_df["new_feature_name"].isin(selected_features)
            ]
            approved_features_with_scores = simulate_importance_scores(approved_features)
            save_to_feature_repository(approved_features_with_scores, transcript_id_inner)

            # Save summary output
            summary_output = approved_features_with_scores[
                [
                    "transcript_id",
                    "generated_modus_operandi",
                    "new_feature_name",
                    "description",
                    "importance_score",
                ]
            ].rename(columns={"generated_modus_operandi": "modus_operandi"})
            summary_output_path = output_dir / f"{transcript_id_inner}_summary.csv"
            summary_output.to_csv(summary_output_path, index=False)

            success_banner(
                "Approved features saved to the repository and summary exported."
            )
            st.caption(f"Summary saved to: `{summary_output_path}`")

            st.session_state.selected_features[transcript_id_inner] = []
            st.session_state.form_submitted[transcript_id_inner] = True
        else:
            warning_banner("Please select at least one feature to approve.")
            logger.info(f"No features selected for approval in {transcript_id_inner}")

    with tab_features:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Feature suggestions & approval")

        st.markdown(
            "Review LLM-suggested features, map to raw variables, and approve what should move towards production."
        )

        # Recommended features table
        st.markdown("#### Recommended features from Gemini")
        st.dataframe(
            features_df[
                [
                    "new_feature_name",
                    "description",
                    "required_raw_variables",
                    "remark",
                ]
            ],
            use_container_width=True,
        )

        st.markdown("---")

        # Importance scores
        st.markdown("#### Importance scores (simulated)")
        if not features_with_scores.empty:
            st.dataframe(
                features_with_scores[["new_feature_name", "importance_score"]],
                use_container_width=True,
            )

            all_features_scores = pd.concat(
                [
                    existing_features_df[["feature_name", "importance_score"]]
                    .rename(columns={"feature_name": "new_feature_name"})
                    if not existing_features_df.empty
                    else pd.DataFrame(
                        columns=["new_feature_name", "importance_score"]
                    ),
                    features_with_scores[["new_feature_name", "importance_score"]],
                ],
                ignore_index=True,
            )

            st.markdown("#### Portfolio view – existing + recommended")
            st.dataframe(all_features_scores, use_container_width=True)

        st.markdown("---")

        # SHAP plots
        st.markdown("#### Model impact – SHAP before vs after")
        col_before, col_after = st.columns(2)
        with col_before:
            st.markdown("**Before adding recommended features**")
            st.image(
                shap_plots["before"],
                caption="SHAP summary · existing features",
                use_column_width=True,
            )
        with col_after:
            st.markdown("**After adding recommended features**")
            st.image(
                shap_plots["after"],
                caption="SHAP summary · with recommended features",
                use_column_width=True,
            )

        st.markdown("---")

        # Approvals form
        st.markdown("#### Approve features for repository")
        if (
            not st.session_state.features_df.empty
            and "new_feature_name" in st.session_state.features_df.columns
        ):
            with st.form(key=f"approve_features_form_{transcript_id}"):
                selected_features = st.multiselect(
                    "Select features to approve:",
                    options=st.session_state.features_df["new_feature_name"].tolist(),
                    default=st.session_state.selected_features.get(transcript_id, []),
                    key=f"multiselect_{transcript_id}",
                )
                st.form_submit_button(
                    "Approve selected features",
                    on_click=handle_form_submission,
                    args=(transcript_id,),
                )
        else:
            warning_banner(
                "Cannot approve features: no valid features available. "
                "Check dashboard.log for details."
            )

        st.markdown("</div>", unsafe_allow_html=True)

else:
    warning_banner("Upload a transcript (.txt) from the sidebar to start the analysis.")
