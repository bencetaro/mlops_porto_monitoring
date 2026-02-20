# MLOps Inference & Monitoring Pipeline

A complete MLOps project featuring:
- Model training
- REST inference API
- Streamlit client
- Prometheus metrics
- Grafana dashboards
- Dockerized microservices

This project demonstrates a production-style ML deployment. 
The goal is to simulate a realistic ML production environment with observability and reproducibility in mind.

---

## Project Structure

        root/
        ├── data/
        | ├── inference/
        │ | ├── test_inference_api_1.csv
        │ | └── test_inference_api_2.csv
        | └── training/
        |  └── raw/
        |   ├── test.csv.zip
        |   └── train.csv.zip
        ├── docker/
        | ├── Dockerfile.inference
        | ├── Dockerfile.train
        | ├── Dockerfile.streamlit
        | └── run_train_pipeline.sh
        ├── images/...
        ├── monitoring/
        │ ├── prometheus.yml
        │ └── grafana/
        │  ├── datasource.yml
        │  ├── provider.yml
        │  └── dashboards/
        |   └── InferenceDashb.json
        ├── output/...
        ├── src/
        | ├── inference/
        | │ ├── api_service.py
        | │ ├── helpers.py
        | │ ├── client.py
        | │ ├── schemas.py
        | │ └── ui/
        | |  ├── inference_ui.py
        | |  ├── model_comparison.py
        | |  └── training_ui.py
        | └── training/
        |  ├── train_model.py
        |  ├── data_prep.py
        |  ├── helpers.py
        |  └── train_config.yml
        ├── docker-compose.yml
        └── requirements.txt

---

## Web Application for Inference

### Interface Structure

The application consists of three main pages:

- **Inference Page:**
  Supports both single prediction and batch prediction workflows.

- **Training Page:**
  Displays training statistics, performance metrics, and model insights.

- **Model Comparison Page:**
  Provides a structured tabular comparison between available models.

### Backend Architecture

FastAPI acts as the orchestration layer between the frontend and the models by:

- Serving prediction endpoints
- Managing batch inference requests
- Providing structured API responses for seamless frontend integration
- Extracting model and system-level metrics for observability

---

## Monitoring Parameters

### Application-Level Metrics

| Metric | Type | Purpose |
|--------|------|----------|
| `inference_requests_total` | Counter | Total number of inference requests |
| `inference_errors_total` | Counter | Number of failed predictions |
| `inference_latency_seconds`  | Histogram | Inference latency in seconds |
| `page_views_total` | Counter | Web traffic monitoring |
| `inference_prediction_value` | Histogram | Distribution of predicted values |
| `batch_size` | Histogram | Measure batch input sizes |

### Infrastructure-Level Metrics

Collected by Node Exporter:

- CPU usage
- Memory usage
- Disk I/O
- Process statistics

---

## Inference API Snapshots

![Inference Batch Prediciton](https://github.com/bencetaro/mlops_porto_monitoring/blob/master/images/mlops_infer.png)
![Inference Batch Prediciton](https://github.com/bencetaro/mlops_porto_monitoring/blob/master/images/mlops_infer_preds_2.png)
![Training Results](https://github.com/bencetaro/mlops_porto_monitoring/blob/master/images/mlops_train1.png)
![Training Results](https://github.com/bencetaro/mlops_porto_monitoring/blob/master/images/mlops_train2.png)
![Model Comparison](https://github.com/bencetaro/mlops_porto_monitoring/blob/master/images/mlops_comp.png)

---

## Grafana Dashboard Snapshots

![Dashboard](https://github.com/bencetaro/mlops_porto_monitoring/blob/master/images/grafana_1.png)
![Dashboard](https://github.com/bencetaro/mlops_porto_monitoring/blob/master/images/grafana_2.png)

