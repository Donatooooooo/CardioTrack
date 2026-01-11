# Reports

This folder contains all the reports generated during the development and testing of the CardioTrack project, including test results, data validation reports, and monitoring information.

## Folder Structure

```
reports/
├── figures/                            <- Monitoring charts and visualizations
├── great_expectations_reports/         <- Data quality validation reports
├── pytest_report/                      <- Unit and behavioral test report
├── deepchecks_data_drift_reports/      <- Deepchecks reports
└── locust_reports/                     <- Locust reports
```

## Unit and behavioral test report

### Behavioral Model Tests

This suite contains **12 tests** focused on validating the ML model behavior. It includes directional tests (verifying that predictions change as expected when features change), invariance tests (ensuring stability under certain transformations), and minimum functionality tests (checking basic model operations). All tests passed successfully.

### Project Tests

The main test suite covers **80 tests** across the entire codebase, including API endpoints, data preprocessing, model training, evaluation, and explainability features. All tests passed successfully.

## Data Validation Reports

Data quality is validated using Great Expectations. The HTML reports in `great_expectations_reports/` show validation results for both the raw and processed datasets, ensuring data integrity throughout the pipeline.

## Monitoring

### Prometheus
Prometheus is used to collect and store application metrics from the CardioTrack API. It enables real-time monitoring of prediction counts and processing latencies with custom metrics and general system health with deafult metrics.

**Prometheus Configuration:**

| Setting | Value | Rationale |
|---------|-------|-----------|
| Global scrape_interval | 15s | Standard for moderate workloads |
| FastAPI scrape_interval | 5s | Higher granularity for application metrics |
| metrics_path | /metrics | Prometheus standard endpoint |

**Custom Metrics Exposed:**
| Metric | Type | Purpose |
|--------|------|---------|
| cardiotrack_api_predictions_total | Counter | Total number of prediction requests (labels: prediction_type, endpoint) |
| cardiotrack_api_prediction_results_total | Counter | Track predictions by class (labels: prediction_class, endpoint) |
| cardiotrack_api_model_errors_total | Counter | Track model loading or prediction errors (labels: error_type, endpoint) |
| cardiotrack_api_explanations_total | Counter | Total number of explanation requests (labels: status, endpoint) |
| cardiotrack_api_batch_size | Histogram | Distribution of batch prediction sizes (buckets: 1-500) |
| cardiotrack_api_prediction_processing_seconds | Histogram | Identify model bottlenecks (buckets: 0.01s-120s) |

**Default Metrics:**
| Metric | Purpose |
|--------|---------|
| cardiotrack_api_request_size_bytes | Size of incoming HTTP requests in bytes |
| cardiotrack_api_response_size_bytes | Size of outgoing HTTP responses in bytes |
| cardiotrack_api_http_request_duration_seconds | HTTP request latency (buckets: 0.005s-2.5s) |
| cardiotrack_api_http_requests_total | Total HTTP requests count |

### Uptime
The Better Stack dashboard displays real-time uptime monitoring for the CardioTrack API hosted on Hugging Face Spaces. The CardioTrack-Monitoring-Alert monitor performs health checks every 3 minutes and is configured to automatically notify the team via email when incidents occur.
![API Uptime Monitoring](figures/uptime.png)

### Load Testing - Locust 

Two distinct load tests were conducted using Locust to evaluate the behavior of the system under different execution conditions.

#### Load test 1 – Standard load scenario
- Duration: 38 minutes
- Total requests: 9,096
- Failures: 0

This test represents a standard and stable load scenario. Lightweight endpoints for navigation and metadata (/, /cards/*, /model/*) exhibited very low response times (≈15–20 ms on average), confirming high responsiveness.

Prediction endpoints (/predictions and /batch-predictions) showed average latencies around 250 ms, consistent with real-time ML inference.
The explainability endpoint (/explanations) was the most computationally expensive, with an average response time of approximately 3.4 s and peaks up to 21 s, which is expected given the complexity of model explanation processes.

Overall, the system demonstrated high stability, good performance, and balanced behavior under sustained load.
![Locust Monitoring](figures/Locust_graph_1.png)

#### Load Test 2 – Constrained / Cold-Start Scenario
- Duration: 16 minutes
- Total requests: 2,425
- Failures: 0

In this second test, all endpoints exhibited very high response times, with average latencies in the range of 350–430 seconds and peaks close to 15 minutes. This behavior affected both lightweight endpoints and computationally intensive ones.

Despite the extreme latencies, no requests failed, indicating that the system remained functionally stable. The observed performance degradation is consistent with a constrained execution context, from limited computational resources.
![Locust Monitoring](figures/Locust_graph_2.png)

### Data Drift Monitoring - Deepchecks

To ensure the long-term reliability of the CardioTrack predictive model, a dedicated **data drift monitoring process** has been implemented. The goal of this process is to continuously assess whether the statistical properties of incoming production data remain consistent with those observed during training, and to detect distribution shifts that may negatively impact model performance.

Feature-level drift detection is performed using **Deepchecks**, adopting statistical tests selected according to the nature of each feature:
- **Kolmogorov–Smirnov test** for numerical features, used to compare empirical distributions between reference and production data.
- **Cramér’s V** for categorical features, used to quantify changes in categorical distributions.


The drift monitoring process is **automatically executed once per day at 21:00**, a time chosen to minimize interference with peak application traffic, as lower user activity is expected during evening hours.

Two daily drift analyses are reported, illustrating different operating conditions. In the [first run](reports/deepchecks_data_drift_reports/drift_result_2026-01-10_11-10-05.json), conducted on a larger production sample, multiple features exceeded the configured drift threshold (0.2), resulting in a relatively high mean drift and indicating a significant shift in the data distribution compared to the reference dataset. This scenario represents a potential risk for model reliability and highlights the importance of continuous monitoring.

In the [second run](reports/deepchecks_data_drift_reports/drift_result_2026-01-10_16-30-14.json), performed later the same day on a much smaller production sample, the overall drift was significantly lower, with only a single numerical feature slightly exceeding the threshold and all categorical features remaining stable. The reduced mean drift indicates a largely consistent data distribution with respect to the reference.

Together, these results demonstrate that the adopted drift monitoring strategy is capable of capturing both significant and negligible distribution shifts, supporting informed MLOps decisions such as alerting, investigation, or model retraining when required.

### Grafana Dashboard

A centralized monitoring dashboard has been implemented using **Grafana** to provide real-time visibility into the CardioTrack API health and performance. The dashboard focuses on essential metrics only, keeping the visualization lightweight while still providing immediate answers to critical operational questions.

![CardioTrack Grafana Dashboard](figures/grafana_dashboard.png)

*Figure: The Grafana monitoring dashboard developed for CardioTrack, captured during a load testing session. The top row shows service health, success rate, and prediction distribution; the bottom row displays request throughput, error rate, and model inference latency.*

#### Dashboard Architecture

The monitoring stack consists of Prometheus as the metrics backend and Grafana as the visualization layer. Metrics are divided into two categories:

- **Infrastructure metrics**: automatically collected by `prometheus-fastapi-instrumentator`, covering HTTP request counts, response latencies, and status codes.
- **Custom ML metrics**: specifically implemented to track model behavior, including prediction distributions and processing times.

This separation allows operators to distinguish between infrastructure issues (slow network, server overload) and application-level problems.

#### Panel Overview

| Panel | Metric Type | Purpose |
|-------|-------------|---------|
| Services UP | Infrastructure | Verify that FastAPI and Prometheus services are reachable |
| Success Rate | Infrastructure | Percentage of successful requests (2xx responses) |
| Prediction Distribution | Custom | Distribution of model outputs across classes (Healthy/Heart Disease) |
| Request by Endpoints | Infrastructure | Traffic breakdown by API endpoint |
| Total Requests | Infrastructure | Real-time request throughput (req/s) |
| Error Rate | Infrastructure | Rate of client (4xx) and server (5xx) errors |
| Prediction Latency | Custom | P95 latency of model inference, excluding HTTP overhead |



