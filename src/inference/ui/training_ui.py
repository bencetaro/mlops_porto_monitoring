import streamlit as st
import pandas as pd
import os
from src.inference.helpers import list_model_meta, load_meta, get_plot_path

def show_training_ui():
    st.title("Training Dashboard")

    metas = list_model_meta()
    if not metas:
        st.warning("No meta found")
        return

    model_names = [m["model_name"] for m in metas]
    selected = st.selectbox("Select model", model_names)

    meta = load_meta(selected)
    if not meta:
        st.error("Meta missing")
        return

    st.subheader("Training Metadata")
    df_meta = pd.DataFrame(meta.items(), columns=["Attribute", "Value"])
    st.table(df_meta)

    st.subheader("Feature Importance Plot")
    fi = get_plot_path(selected, "fi")
    if os.path.exists(fi):
        st.image(fi)
    else:
        st.info("FI plot not found")

    st.subheader("Mutual Information Plot")
    mi = get_plot_path(selected, "mi")
    if os.path.exists(mi):
        st.image(mi)
    else:
        st.info("MI plot not found")

    st.subheader("Confusion Matrix Plot")
    cm = get_plot_path(selected, "cm")
    if os.path.exists(cm):
        st.image(cm)
    else:
        st.info("CM plot not found")
