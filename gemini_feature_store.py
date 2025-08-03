import json
import pandas as pd
import hashlib
from datetime import timedelta
from feast import FeatureStore, Entity, FeatureView, Field, FileSource
from feast.types import Float32, String
import os
import shap
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------
# A) AUTO-GENERATE FEATURES
# -----------------------------
"""
def load_llm_features(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def generate_feature_code(feature_row):
    logic = f"""
#def compute_{feature_row['new_feature_name']}(df):
    # Required vars: {feature_row['required_raw_variables']}
    # Logic: {feature_row['description']}
    # This is a placeholder function - implement actual logic.
    df['{feature_row['new_feature_name']}'] = 0.0
#    return df
"""
    return logic

def save_feature_logic_scripts(df, output_dir="generated_features"):
    os.makedirs(output_dir, exist_ok=True)
    for _, row in df.iterrows():
        feature_name = row['new_feature_name']
        code = generate_feature_code(row)
        with open(os.path.join(output_dir, f"{feature_name}.py"), 'w') as f:
            f.write(code)
"""

# -----------------------------
# B) FEATURE RANKING (SHAP)
# -----------------------------

def rank_features_with_shap(df, feature_data_path, target_column):
    df_list = []
    feature_names = df['new_feature_name'].tolist()

    for feature_name in feature_names:
        feature_file = os.path.join(feature_data_path, f"{feature_name}.parquet")
        if os.path.exists(feature_file):
            df_feat = pd.read_parquet(feature_file)
            if target_column in df_feat.columns:
                X = df_feat[[feature_name]].fillna(0)
                y = df_feat[target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                model = RandomForestClassifier()
                model.fit(X_train, y_train)

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)

                importance = abs(shap_values[1]).mean() if isinstance(shap_values, list) else abs(shap_values).mean()
                df.loc[df['new_feature_name'] == feature_name, 'importance_score'] = importance
            else:
                df.loc[df['new_feature_name'] == feature_name, 'importance_score'] = 0
        else:
            df.loc[df['new_feature_name'] == feature_name, 'importance_score'] = 0

    df.sort_values(by='importance_score', ascending=False, inplace=True)
    return df


# -----------------------------
# C) FEAST REGISTRATION
# -----------------------------

def register_features_in_feast(df, feature_store_path="my_feature_repo"):
    store = FeatureStore(repo_path=feature_store_path)
    user = Entity(name="user_id", join_keys=["user_id"])
    store.apply([user])

    for _, row in df.iterrows():
        feature_name = row['new_feature_name']
        feature_view = FeatureView(
            name=feature_name,
            entities=["user_id"],
            ttl=timedelta(days=1),
            schema=[
                Field(name=feature_name, dtype=Float32),
            ],
            source=FileSource(
                path=f"data/{feature_name}.parquet",
                event_timestamp_column="event_time",
            ),
        )
        store.apply([feature_view])


# -----------------------------
# D) STREAMLIT UI
# -----------------------------

def launch_streamlit_ui(df):
    st.title("🔍 AI-Recommended Fraud Features")

    st.subheader("Ranked Features by SHAP Importance")
    st.dataframe(df[['new_feature_name', 'importance_score', 'description']])

    st.subheader("Feature Descriptions")
    selected = st.selectbox("Choose a Feature", df['new_feature_name'])
    st.write(df[df['new_feature_name'] == selected]['description'].values[0])


# -----------------------------
# MAIN PIPELINE
# -----------------------------

def main():
    llm_features_path = "gemini_feature_recommendations.json"
    feature_data_path = "data"
    target_column = "is_fraud"  # Must exist in each .parquet for SHAP ranking

    df = load_llm_features(llm_features_path)
    save_feature_logic_scripts(df)

    df = rank_features_with_shap(df, feature_data_path, target_column)
    print("Top recommended features:\n", df[['new_feature_name', 'importance_score']].head())

    register_features_in_feast(df)
    launch_streamlit_ui(df)


if __name__ == "__main__":
    main()
