import os
import pandas as pd
import mlflow

MODEL_DIR = "/app/model"

try:
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"Failed to load model from {MODEL_DIR}: {e}")

try:
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
    with open(feature_file) as f:
        feature_cols = [col.strip() for col in f if col.strip()]
        print(f"Loaded {len(feature_file)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

binary_map = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1}
}

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame):
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0)

    for c, mapping in binary_map.items():
        if c in df.columns:
            df[c] = (df[c]
                     .astype(str)
                     .str.strip()
                     .map(mapping)
                     .fillna(0)
                     .astype(int))

    object_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in binary_map]
    if object_cols:
        df = pd.get_dummies(df, columns = object_cols, drop_first = True)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    df = df.reindex(columns = feature_cols, fill_value = 0)

    return df

def predict(input_dict: dict):
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    try:
        preds = model.predict(df_enc)
        
        if hasattr(preds, "tolist"):
            preds = preds.tolist()

        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds

    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    if result == 1:
        return "Likely to churn"
    else:
        return "Not likely to churn"
