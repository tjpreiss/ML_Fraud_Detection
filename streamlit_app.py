import os
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import boto3
import sagemaker
import shap
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection — IEEE-CIS",
    page_icon="🔍",
    layout="wide"
)

# ─── Constants ────────────────────────────────────────────────────────────────
ENDPOINT_NAME = "fraud-detection-v2"
BUCKET        = "thomas-preiss-s3-bucket"

FEATURE_COLS = [
    'ProductCD', 'card1', 'card3', 'card6', 'C5', 'C7', 'C8', 'C9',
    'C10', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D10', 'D15',
    'M6', 'M8', 'id_01', 'id_02', 'id_05', 'id_06', 'id_15', 'id_17',
    'id_19', 'id_35', 'id_38', 'DeviceType', 'log_C1', 'log_C10',
    'log_C11', 'log_C12', 'log_C13', 'log_C14', 'log_C2', 'pca_0',
    'pca_1', 'pca_2', 'pca_5', 'pca_7', 'pca_9', 'pca_11', 'pca_12',
    'pca_18', 'pca_26', 'pca_37', 'pca_38', 'pca_41'
]

# Median defaults (from training data)
DEFAULTS = {
    'ProductCD': 4.0,    'card1': 9500.0,   'card3': 150.0,   'card6': 1.0,
    'C5': 0.0,           'C7': 0.0,         'C8': 0.0,        'C9': 1.0,
    'C10': 0.0,          'C12': 0.0,        'C13': 3.0,       'C14': 1.0,
    'D1': 10.0,          'D2': 115.0,       'D3': 11.0,       'D4': 88.0,
    'D10': 29.0,         'D15': 147.0,      'M6': 0.0,        'M8': 0.0,
    'id_01': -5.0,       'id_02': 104708.0, 'id_05': 0.0,     'id_06': 0.0,
    'id_15': 1.0,        'id_17': 166.0,    'id_19': 352.0,   'id_35': 1.0,
    'id_38': 1.0,        'DeviceType': 0.0, 'log_C1': 0.6931, 'log_C10': 0.0,
    'log_C11': 0.6931,   'log_C12': 0.0,   'log_C13': 1.3863,'log_C14': 0.6931,
    'log_C2': 0.6931,    'pca_0': -1.2839, 'pca_1': -1.0505, 'pca_2': 0.2134,
    'pca_5': -0.3182,    'pca_7': 0.0251,  'pca_9': -0.1135, 'pca_11': -0.2778,
    'pca_12': -0.0133,   'pca_18': 0.0099, 'pca_26': 0.0138, 'pca_37': -0.0353,
    'pca_38': 0.0038,    'pca_41': -0.0187
}

# User-friendly labels for the key inputs shown in the form
KEY_INPUTS = {
    'ProductCD' : {'label': 'Product Code (0-4)',          'min': 0.0,   'max': 4.0,      'step': 1.0},
    'card1'     : {'label': 'Card 1 ID',                   'min': 0.0,   'max': 18396.0,  'step': 1.0},
    'card3'     : {'label': 'Card 3',                      'min': 100.0, 'max': 231.0,    'step': 1.0},
    'card6'     : {'label': 'Card Type (0=credit,1=debit)','min': 0.0,   'max': 1.0,      'step': 1.0},
    'C9'        : {'label': 'C9 Count',                    'min': 0.0,   'max': 200.0,    'step': 1.0},
    'C13'       : {'label': 'C13 Count',                   'min': 0.0,   'max': 2720.0,   'step': 1.0},
    'D1'        : {'label': 'Days Since Last Txn (D1)',    'min': 0.0,   'max': 640.0,    'step': 1.0},
    'D4'        : {'label': 'Time Delta D4',               'min': 0.0,   'max': 869.0,    'step': 1.0},
    'D10'       : {'label': 'Time Delta D10',              'min': 0.0,   'max': 820.0,    'step': 1.0},
    'id_02'     : {'label': 'Identity Score (id_02)',      'min': 0.0,   'max': 500000.0, 'step': 100.0},
}

# ─── AWS Session ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_aws_session(key_id, secret, token):
    return boto3.Session(
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        aws_session_token=token,
        region_name='us-east-1'
    )

@st.cache_resource
def get_sagemaker_session(_boto_session):
    return sagemaker.Session(boto_session=_boto_session)

# ─── Load SHAP explainer from S3 ──────────────────────────────────────────────
@st.cache_resource
def load_shap_explainer(_boto_session, bucket):
    s3 = _boto_session.client('s3')
    local = '/tmp/shap_explainer.joblib'
    if not os.path.exists(local):
        s3.download_file(Bucket=bucket, Key='fraud-detection/model-v2/shap_explainer.joblib', Filename=local)
    return joblib.load(local)

# ─── Prediction via SageMaker endpoint ───────────────────────────────────────
def call_endpoint(input_dict, sm_session):
    predictor = Predictor(
        endpoint_name=ENDPOINT_NAME,
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )
    payload = json.dumps({"inputs": input_dict})
    try:
        result = predictor.predict(payload, initial_args={"ContentType": "application/json"})
        pred  = result['predictions'][0]
        proba = result['probabilities'][0]
        return pred, proba, None
    except Exception as e:
        return None, None, str(e)

# ─── SHAP waterfall plot ──────────────────────────────────────────────────────
def show_shap(input_dict, explainer):
    X = pd.DataFrame([input_dict]).reindex(columns=FEATURE_COLS, fill_value=0)
    shap_vals = explainer.shap_values(X)
    exp = shap.Explanation(
        values=shap_vals[0],
        base_values=explainer.expected_value,
        data=X.iloc[0].values,
        feature_names=FEATURE_COLS
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(exp, show=False)
    st.pyplot(fig)
    plt.close()

    top_feat = pd.Series(np.abs(shap_vals[0]), index=FEATURE_COLS).idxmax()
    st.info(f"**Key Driver:** The most influential feature for this prediction was **{top_feat}**.")

# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("🔍 IEEE-CIS Fraud Detection")
st.markdown("Enter transaction details below. All remaining features default to training-data medians.")

# Sidebar — AWS credentials
with st.sidebar:
    st.header("⚙️ AWS Configuration")
    st.markdown("Credentials are loaded from Streamlit secrets.")
    st.markdown(f"**Endpoint:** `{ENDPOINT_NAME}`")
    st.markdown(f"**Bucket:** `{BUCKET}`")
    st.divider()
    st.caption("ML Project — Spring 2026")

# Load credentials from st.secrets
try:
    aws_id     = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
    aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
    aws_token  = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
except Exception:
    st.error("AWS credentials not found in Streamlit secrets. Add them to `.streamlit/secrets.toml`.")
    st.stop()

boto_session = get_aws_session(aws_id, aws_secret, aws_token)
sm_session   = get_sagemaker_session(boto_session)

# ─── Input Form ───────────────────────────────────────────────────────────────
st.subheader("Transaction Inputs")
with st.form("fraud_form"):
    cols = st.columns(2)
    user_vals = {}
    for i, (feat, cfg) in enumerate(KEY_INPUTS.items()):
        with cols[i % 2]:
            user_vals[feat] = st.number_input(
                label     = cfg['label'],
                min_value = float(cfg['min']),
                max_value = float(cfg['max']),
                value     = float(DEFAULTS[feat]),
                step      = float(cfg['step'])
            )

    submitted = st.form_submit_button("🚀 Run Fraud Prediction", use_container_width=True)

# ─── Prediction ───────────────────────────────────────────────────────────────
if submitted:
    # Build full feature dict: user inputs override defaults
    input_dict = {**DEFAULTS, **user_vals}

    with st.spinner("Calling SageMaker endpoint..."):
        pred, proba, err = call_endpoint(input_dict, sm_session)

    if err:
        st.error(f"Endpoint error: {err}")
        st.stop()

    # Result display
    st.divider()
    col1, col2, col3 = st.columns(3)
    label = "🚨 FRAUD" if pred == 1 else "✅ LEGITIMATE"
    color = "red" if pred == 1 else "green"

    with col1:
        st.metric("Prediction", label)
    with col2:
        st.metric("Fraud Probability", f"{proba*100:.2f}%")
    with col3:
        risk = "HIGH" if proba > 0.7 else "MEDIUM" if proba > 0.3 else "LOW"
        st.metric("Risk Level", risk)

    # Probability gauge bar
    st.progress(float(proba), text=f"Fraud probability: {proba*100:.1f}%")

    # ─── SHAP Explainability ──────────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 Decision Explanation (SHAP)")
    with st.spinner("Computing SHAP values..."):
        try:
            explainer = load_shap_explainer(boto_session, BUCKET)
            show_shap(input_dict, explainer)
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

    # ─── Feature summary table ────────────────────────────────────────────────
    with st.expander("📋 Full Feature Vector Sent to Model"):
        st.dataframe(
            pd.DataFrame([input_dict]).T.rename(columns={0: 'Value'}),
            use_container_width=True
        )
