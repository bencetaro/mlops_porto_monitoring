import streamlit as st
import pandas as pd
from src.inference.helpers import list_model_meta
import plotly.express as px

def plot_auc_chart(df):
    df = df.sort_values("auc")
    fig = px.bar(
        df,
        x="auc",
        y="model_name",
        orientation="h",
        title="Model AUC Comparison",
        text="auc"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, width="stretch")

def show_model_comparison():
    st.title("Model Comparison")

    metas = list_model_meta()
    if not metas:
        st.warning("No meta found")
        return

    model_names = [m["model_name"] for m in metas]
    selected = st.multiselect("Which models?", model_names, default=model_names)

    all_keys = set()
    for m in metas:
        all_keys |= set(m.keys())
    all_keys.discard("model_name")
    all_keys = sorted(list(all_keys))

    attributes = st.multiselect("Which attributes to compare?", all_keys, default=["auc", "train_rows", "val_rows"])

    if not selected or not attributes:
        st.info("Select models and attributes")
        return

    rows = []
    for m in metas:
        if m["model_name"] not in selected:
            continue
        row = {"model_name": m["model_name"]}
        for attr in attributes:
            v = m.get(attr, None)
            row[attr] = str(v)  # flatten if dict
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df)

    if "auc" in df.columns:
        st.subheader("AUC Comparison Chart")
        #st.bar_chart(df.set_index("model_name")["auc"])
        plot_auc_chart(df)
