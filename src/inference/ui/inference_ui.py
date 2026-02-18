import streamlit as st
import requests
import pandas as pd
import time

def show_inference_ui():
    st.title("MLOps Inference UI")

    # Sidebar settings
    st.sidebar.header("Settings")

    api_base_url = st.sidebar.text_input(
        "Inference API base URL",
        value="http://inference-api:8000"
    )

    try:
        models = requests.get(f"{api_base_url}/models", timeout=2).json()
    except:
        models = []

    if not models:
        st.sidebar.warning("No models found.")
        selected_model = "default"
    else:
        selected_model = st.sidebar.selectbox(
            "Model version",
            ["default"] + models
        )

    mode = st.sidebar.selectbox(
        "Inference mode",
        [
            "Single inference",
            "Batch inference"
        ]
    )

    # Health check
    st.sidebar.subheader("API status")

    try:
        r = requests.get(f"{api_base_url}/health", timeout=2)
        if r.status_code == 200:
            st.sidebar.success("API healthy")
        else:
            st.sidebar.error("API error")
    except:
        st.sidebar.error("API unreachable")

    st.divider()

    # Single Inference
    if mode == "Single inference":
        st.header("Single record inference")

        with st.form("single_form"):
            record_id = st.number_input("id", min_value=0, step=1)

            st.subheader("Features")

            ps_ind_01 = st.number_input("ps_ind_01", value=0)
            ps_ind_02_cat = st.number_input("ps_ind_02_cat", value=0)
            ps_ind_03 = st.number_input("ps_ind_03", value=0)
            ps_car_01_cat = st.number_input("ps_car_01_cat", value=0)
            # ... and more, depends on models feature count
            # of course this wont work with current models

            submitted = st.form_submit_button("Predict")

        if submitted:
            payload = {
                "id": int(record_id),
                "features": {
                    "ps_ind_01": ps_ind_01,
                    "ps_ind_02_cat": ps_ind_02_cat,
                    "ps_ind_03": ps_ind_03,
                    "ps_car_01_cat": ps_car_01_cat,
                }
            }

            try:
                start = time.time()
                r = requests.post(
                    f"{api_base_url}/predict?model_name={selected_model}",
                    json=payload
                )
                latency = time.time() - start

                if r.status_code == 200:
                    res = r.json()
                    st.success("Prediction successful")
                    st.metric("Prediction", round(res["prediction"], 4))
                    st.caption(f"Latency: {latency:.3f}s")
                else:
                    st.error(r.text)

            except Exception as e:
                st.error(str(e))

    # Batch Inference
    else:
        st.header("Batch inference")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            check_df = df.copy()
            has_id = "id" in check_df.columns

            if has_id:
                id_ = df["id"].reset_index(drop=True)
                df = df.drop(columns=["id"])

            st.subheader("Input preview")
            st.dataframe(df.head().astype(str))

            if st.button("Run batch inference"):

                payload = df.to_dict(orient="records")

                try:
                    start = time.time()
                    r = requests.post(
                        f"{api_base_url}/predict/batch?model_name={selected_model}",
                        json=payload
                    )
                    latency = time.time() - start

                    if r.status_code == 200:
                        preds = pd.DataFrame(r.json())
                        preds["prediction"] = preds["prediction"].astype(float)
                        
                        if has_id:
                            result = pd.concat([id_, preds], axis=1)
                        else:
                            result = pd.concat([df.reset_index(drop=True), preds], axis=1)

                        st.success("Batch prediction successful")
                        st.caption(f"Latency: {latency:.3f}s")

                        st.subheader("Predictions")
                        st.dataframe(result.head().astype(str))

                        csv = result.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download CSV",
                            csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(r.text)

                except Exception as e:
                    st.error(str(e))
