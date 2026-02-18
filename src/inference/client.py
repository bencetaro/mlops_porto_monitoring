import streamlit as st

from src.inference.ui.inference_ui import show_inference_ui
from src.inference.ui.training_ui import show_training_ui
from src.inference.ui.model_comparison import show_model_comparison

st.set_page_config(page_title="MLOps UI", layout="wide")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Inference UI",
        "Training UI",
        "Model Comparison"
    ]
)

if page == "Inference UI":
    show_inference_ui()

elif page == "Training UI":
    show_training_ui()

elif page == "Model Comparison":
    show_model_comparison()
