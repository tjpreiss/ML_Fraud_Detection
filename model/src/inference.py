import joblib
import os
import json
import numpy as np
import pandas as pd

def model_fn(model_dir):
    pipeline = joblib.load(os.path.join(model_dir, "fraud_pipeline.joblib"))
    return pipeline

def input_fn(request_body, content_type):
    if content_type == "application/json":
        data = json.loads(request_body)
        rows = data["inputs"] if isinstance(data["inputs"], list) else [data["inputs"]]
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_df, pipeline):
    model        = pipeline["model"]
    feature_cols = pipeline["feature_cols"]
    X = input_df.reindex(columns=feature_cols, fill_value=0)
    proba = model.predict_proba(X)[:, 1]
    pred  = model.predict(X)
    return {"predictions": pred.tolist(), "probabilities": proba.tolist()}

def output_fn(prediction, content_type):
    return json.dumps(prediction), "application/json"
