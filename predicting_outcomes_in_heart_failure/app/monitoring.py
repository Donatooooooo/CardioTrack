import os

from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator, metrics

NAMESPACE = os.environ.get("METRICS_NAMESPACE", "cardiotrack")
SUBSYSTEM = os.environ.get("METRICS_SUBSYSTEM", "api")

instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    inprogress_name="fastapi_inprogress",
    inprogress_labels=True,
)

instrumentator.add(
    metrics.request_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

instrumentator.add(
    metrics.response_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

instrumentator.add(
    metrics.latency(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
    )
)

instrumentator.add(
    metrics.requests(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

prediction_counter = Counter(
    name=f"{NAMESPACE}_{SUBSYSTEM}_predictions_total",
    documentation="Total number of prediction requests",
    labelnames=["prediction_type", "endpoint"],
)

prediction_result_counter = Counter(
    name=f"{NAMESPACE}_{SUBSYSTEM}_prediction_results_total",
    documentation="Count of prediction results by class",
    labelnames=["prediction_class", "endpoint"],
)

model_error_counter = Counter(
    name=f"{NAMESPACE}_{SUBSYSTEM}_model_errors_total",
    documentation="Total number of model loading or prediction errors",
    labelnames=["error_type", "endpoint"],
)

explanation_counter = Counter(
    name=f"{NAMESPACE}_{SUBSYSTEM}_explanations_total",
    documentation="Total number of explanation requests",
    labelnames=["status", "endpoint"],
)

batch_size_histogram = Histogram(
    name=f"{NAMESPACE}_{SUBSYSTEM}_batch_size",
    documentation="Distribution of batch prediction sizes",
    labelnames=["endpoint"],
    buckets=[1, 5, 10, 20, 50, 100, 200, 500],
)

prediction_processing_time = Histogram(
    name=f"{NAMESPACE}_{SUBSYSTEM}_prediction_processing_seconds",
    documentation="Time spent on prediction processing (excluding HTTP overhead)",
    labelnames=["prediction_type", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)
