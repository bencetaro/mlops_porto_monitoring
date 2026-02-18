import os
import joblib
import json
import numpy as np
import pandas as pd

def list_model_meta(output_dir="/output"):
    params_dir = os.path.join(output_dir, "params")
    if not os.path.exists(params_dir):
        return []

    metas = []
    for f in os.listdir(params_dir):
        if f.startswith("meta_") and f.endswith(".json"):
            with open(os.path.join(params_dir, f)) as jf:
                meta = json.load(jf)
                metas.append(meta)

    return metas

def load_meta(model_name, output_dir="/output"):
    path = os.path.join(output_dir, "params", f"meta_{model_name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def get_plot_path(model_name, plot_type, output_dir="/output"):
    # plot_type: fi / mi / cm
    return os.path.join(output_dir, "plots", f"{plot_type}_{model_name}.png")

def inference_preprocessing(df: pd.DataFrame, preprocessor_path: str):
    assert os.path.exists(preprocessor_path), "preprocessors.pkl must exist in input directory for inference mode."
    num_imputer, cat_imputer, bin_imputer, scaler, le_encoder, cols_to_drop, bin_cols, cat_cols, num_cols = joblib.load(preprocessor_path)
    df = df.copy()

    # General preprocessing
    df.drop("id", axis=1, inplace=True, errors="ignore")
    numericals = [c for c in df.columns if not (c.endswith("cat") or c.endswith("bin") or c.endswith("target"))]
    df.rename(columns={c: c + "_num" for c in numericals}, inplace=True)
    df.replace(-1, np.nan, inplace=True)

    # Drop any columns unused in training
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors="ignore")

    # Align unseen data exactly to training schema
    df = df.reindex(columns=num_cols + cat_cols + bin_cols, fill_value=np.nan)

    # Apply imputers/scalers/encoders
    df[num_cols] = num_imputer.transform(df[num_cols])
    df[cat_cols] = cat_imputer.transform(df[cat_cols])
    df[bin_cols] = np.round(bin_imputer.transform(df[bin_cols]))
    df[num_cols] = scaler.transform(df[num_cols])
    df[cat_cols] = le_encoder.transform(df[cat_cols])

    return df
