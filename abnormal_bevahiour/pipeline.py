import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics


@component(
    base_image="python:3.9",
    packages_to_install=["elasticsearch==8.12.0", "pandas"]
)
def fetch_traffic_data(
    es_host:         str,
    es_user:         str,
    es_password:     str,
    es_index_prefix: str,
    fetch_days:      int,
    bucket_interval: str,
    output_data:     Output[Dataset],
):
    """Pull 15-min aggregated traffic from Elasticsearch for anomaly detection training."""
    from elasticsearch import Elasticsearch
    import pandas as pd
    from datetime import datetime, timedelta

    if not es_host:
        raise ValueError("es_host is required")
    if not es_user:
        raise ValueError("es_user is required")
    if not es_password:
        raise ValueError("es_password is required")

    es = Elasticsearch(
        es_host,
        basic_auth=(es_user, es_password),
        verify_certs=False,
        request_timeout=120,
    )

    today = datetime.utcnow().date()
    since = today - timedelta(days=fetch_days)

    indices = []
    for offset in range(fetch_days + 1):
        day = today - timedelta(days=offset)
        idx = f"{es_index_prefix}-{day.strftime('%Y.%m.%d')}"
        if es.indices.exists(index=idx):
            indices.append(idx)
            print(f"Found: {idx}")
        else:
            print(f"Missing: {idx} — skipping")

    if not indices:
        raise RuntimeError(f"No indices found for the last {fetch_days} days.")

    query = {
        "size": 0,
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "@timestamp": {
                                "gte": since.isoformat(),
                                "lte": (today + timedelta(days=1)).isoformat(),
                            }
                        }
                    },
                    {"term": {"flow.final": True}}
                ]
            }
        },
        "aggs": {
            "traffic_over_time": {
                "date_histogram": {
                    "field":          "@timestamp",
                    "fixed_interval": bucket_interval,
                    "min_doc_count":  0,
                    "extended_bounds": {
                        "min": since.isoformat(),
                        "max": today.isoformat(),
                    },
                },
                "aggs": {
                    "avg_bytes":     {"avg":         {"field": "network.bytes"}},
                    "total_bytes":   {"sum":         {"field": "network.bytes"}},
                    "total_packets": {"sum":         {"field": "network.packets"}},
                    "unique_flows":  {"cardinality": {"field": "flow.id.keyword"}},
                    "avg_duration":  {"avg":         {"field": "event.duration"}},
                    "src_bytes":     {"sum":         {"field": "source.bytes"}},
                    "dst_bytes":     {"sum":         {"field": "destination.bytes"}},
                }
            }
        }
    }

    resp    = es.search(index=",".join(indices), body=query)
    buckets = resp["aggregations"]["traffic_over_time"]["buckets"]

    if not buckets:
        raise RuntimeError("No aggregation buckets returned — check index field names.")

    records = []
    for b in buckets:
        records.append({
            "timestamp":     b["key_as_string"],
            "avg_bytes":     b["avg_bytes"]["value"]     or 0,
            "total_bytes":   b["total_bytes"]["value"]   or 0,
            "total_packets": b["total_packets"]["value"] or 0,
            "unique_flows":  b["unique_flows"]["value"]  or 0,
            "avg_duration":  b["avg_duration"]["value"]  or 0,
            "src_bytes":     b["src_bytes"]["value"]     or 0,
            "dst_bytes":     b["dst_bytes"]["value"]     or 0,
            "doc_count":     b["doc_count"],
        })

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(output_data.path, index=False)

    print(f"Fetched {len(df)} buckets ({bucket_interval}) across {len(indices)} indices")
    print(f"Range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"avg_bytes stats: median={df['avg_bytes'].median():.0f}, "
          f"p99={df['avg_bytes'].quantile(0.99):.0f}, max={df['avg_bytes'].max():.0f}")


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "boto3"]
)
def preprocess_features(
    input_data:    Input[Dataset],
    output_data:   Output[Dataset],
    s3_endpoint:   str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_bucket:     str,
    history_key:   str,
):
    """Compute derived + temporal features, save history snapshot to S3."""
    import pandas as pd
    import numpy as np
    import boto3, io
    from botocore.client import Config

    df = pd.read_csv(input_data.path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Rows before processing: {len(df)}")

    df["bytes_per_packet"] = df["total_bytes"] / df["total_packets"].replace(0, np.nan)
    df["bytes_per_flow"]   = df["total_bytes"] / df["unique_flows"].replace(0, np.nan)
    df["src_dst_ratio"]    = df["src_bytes"]   / df["dst_bytes"].replace(0, np.nan)

    df[["bytes_per_packet", "bytes_per_flow", "src_dst_ratio"]] = \
        df[["bytes_per_packet", "bytes_per_flow", "src_dst_ratio"]].fillna(0)

    df["hour_of_day"]  = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["minute_of_day"] = df["timestamp"].dt.hour * 4 + df["timestamp"].dt.minute // 15
    df["mod_sin"]      = np.sin(2 * np.pi * df["minute_of_day"] / 96)
    df["mod_cos"]      = np.cos(2 * np.pi * df["minute_of_day"] / 96)

    df["roll_mean_4"]  = df["avg_bytes"].shift(1).rolling(4,  min_periods=1).mean()
    df["roll_mean_16"] = df["avg_bytes"].shift(1).rolling(16, min_periods=1).mean()
    df["roll_std_16"]  = df["avg_bytes"].shift(1).rolling(16, min_periods=1).std()
    df["roll_mean_96"] = df["avg_bytes"].shift(1).rolling(96, min_periods=1).mean()

    df = df.fillna(0)

    print(f"Rows after feature engineering: {len(df)}")
    print(f"Features: {list(df.columns)}")

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        s3.put_object(
            Bucket=s3_bucket,
            Key=history_key,
            Body=buf.getvalue().encode("utf-8"),
            ContentType="text/csv",
        )
        print(f"History snapshot → s3://{s3_bucket}/{history_key}")
    except Exception as e:
        print(f"Warning: could not save history to S3: {e}")

    df.to_csv(output_data.path, index=False)


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "scikit-learn==1.5.2", "joblib==1.4.2"]
)
def train_anomaly_detector(
    input_data:     Input[Dataset],
    model_output:   Output[Model],
    metrics:        Output[Metrics],
    contamination:  float,
    n_estimators:   int,
):
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler
    import joblib, os, json

    df = pd.read_csv(input_data.path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    n_rows = len(df)
    print(f"Training on {n_rows} buckets")

    # Features used by Isolation Forest
    feature_cols = [
        "avg_bytes", "total_packets", "unique_flows",
        "bytes_per_flow", "bytes_per_packet", "src_dst_ratio",
    ]

    log_cols = ["avg_bytes", "total_packets", "unique_flows", "bytes_per_flow"]

    X = df[feature_cols].copy().fillna(0)
    for col in log_cols:
        X[col] = np.log1p(X[col].clip(lower=0))

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    preds  = iso.predict(X_scaled)
    scores = -iso.score_samples(X_scaled)  # higher = more anomalous

    n_anomalies = int((preds == -1).sum())
    anomaly_rate = n_anomalies / n_rows

    print(f"Flagged {n_anomalies} anomalies out of {n_rows} ({anomaly_rate:.2%})")
    print(f"Anomaly score: min={scores.min():.3f}, max={scores.max():.3f}, "
          f"mean={scores.mean():.3f}")

    # Show top anomalies for logs
    df_with_flags = df.copy()
    df_with_flags["iso_score"] = scores
    df_with_flags["iso_flag"]  = (preds == -1).astype(int)
    top = df_with_flags.nlargest(10, "iso_score")[
        ["timestamp", "avg_bytes", "unique_flows", "total_packets", "iso_score"]
    ]
    print("Top 10 anomaly candidates:")
    print(top.to_string(index=False))

    metrics.log_metric("n_buckets",        float(n_rows))
    metrics.log_metric("n_anomalies",      float(n_anomalies))
    metrics.log_metric("anomaly_rate_pct", float(anomaly_rate * 100))
    metrics.log_metric("contamination",    float(contamination))
    metrics.log_metric("score_mean",       float(scores.mean()))
    metrics.log_metric("score_max",        float(scores.max()))

    os.makedirs(model_output.path, exist_ok=True)
    joblib.dump(iso,    os.path.join(model_output.path, "iso_forest.joblib"))
    joblib.dump(scaler, os.path.join(model_output.path, "scaler.joblib"))
    with open(os.path.join(model_output.path, "features.json"), "w") as f:
        json.dump({"features": feature_cols, "log_cols": log_cols}, f)

    print("Model + scaler + features saved ✓")


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "scikit-learn==1.5.2", "joblib==1.4.2", "boto3"]
)
def detect_anomalies_and_store(
    input_data:         Input[Dataset],
    model_input:        Input[Model],
    s3_endpoint:        str,
    s3_access_key:      str,
    s3_secret_key:      str,
    s3_bucket:          str,
    zscore_threshold:   float,
    min_abs_deviation:  float,
    iqr_multiplier:     float,
    rolling_window:     int,
):
    """Run ensemble: Isolation Forest + rolling z-score + rolling IQR, store to S3."""
    import pandas as pd
    import numpy as np
    import joblib, os, json, io
    from datetime import datetime
    import boto3
    from botocore.client import Config

    df = pd.read_csv(input_data.path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Load trained Isolation Forest ---
    iso    = joblib.load(os.path.join(model_input.path, "iso_forest.joblib"))
    scaler = joblib.load(os.path.join(model_input.path, "scaler.joblib"))
    with open(os.path.join(model_input.path, "features.json")) as f:
        feat_config  = json.load(f)
    feature_cols = feat_config["features"]
    log_cols     = feat_config["log_cols"]

    # --- Detector 1: Isolation Forest ---
    X = df[feature_cols].copy().fillna(0)
    for col in log_cols:
        X[col] = np.log1p(X[col].clip(lower=0))
    X_scaled = scaler.transform(X)

    df["iso_score"]   = -iso.score_samples(X_scaled)
    df["anomaly_iso"] = (iso.predict(X_scaled) == -1).astype(int)

    # --- Detector 2: Rolling z-score on avg_bytes ---
    roll_mean = df["avg_bytes"].shift(1).rolling(rolling_window, min_periods=8).mean()
    roll_std  = df["avg_bytes"].shift(1).rolling(rolling_window, min_periods=8).std()
    df["zscore"] = (df["avg_bytes"] - roll_mean) / roll_std.replace(0, np.nan)
    df["anomaly_zscore"] = (
        (df["zscore"].abs() > zscore_threshold)
        & ((df["avg_bytes"] - roll_mean).abs() > min_abs_deviation)
    ).fillna(False).astype(int)

    # --- Detector 3: Rolling IQR ---
    q1 = df["avg_bytes"].shift(1).rolling(rolling_window, min_periods=8).quantile(0.25)
    q3 = df["avg_bytes"].shift(1).rolling(rolling_window, min_periods=8).quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + iqr_multiplier * iqr
    df["anomaly_iqr"] = (df["avg_bytes"] > upper_bound).fillna(False).astype(int)

    # --- Ensemble ---
    df["anomaly_count"]     = df[["anomaly_iso", "anomaly_zscore", "anomaly_iqr"]].sum(axis=1)
    df["anomaly_consensus"] = (df["anomaly_count"] >= 2).astype(int)

    def severity(row):
        if row["anomaly_count"] == 0:
            return "normal"
        if row["anomaly_count"] == 3:
            return "critical"
        if row["anomaly_count"] == 2:
            return "high"
        return "low"

    df["severity"] = df.apply(severity, axis=1)

    n_total     = len(df)
    n_iso       = int(df["anomaly_iso"].sum())
    n_zscore    = int(df["anomaly_zscore"].sum())
    n_iqr       = int(df["anomaly_iqr"].sum())
    n_consensus = int(df["anomaly_consensus"].sum())

    print(f"Total buckets:       {n_total}")
    print(f"Isolation Forest:    {n_iso}")
    print(f"Z-score:             {n_zscore}")
    print(f"IQR:                 {n_iqr}")
    print(f"Consensus (≥2):      {n_consensus}")
    print(f"Critical (all 3):    {int((df['severity'] == 'critical').sum())}")

    agreement = df["anomaly_count"].value_counts().sort_index()
    print(f"Agreement distribution:\n{agreement.to_string()}")

    # Print top consensus anomalies
    consensus = df[df["anomaly_consensus"] == 1].copy()
    if len(consensus) > 0:
        print(f"\nConsensus anomalies (severity ≥ high):")
        cols = ["timestamp", "avg_bytes", "unique_flows", "total_packets",
                "iso_score", "zscore", "anomaly_count", "severity"]
        print(consensus[cols].to_string(index=False))
    else:
        print("No consensus anomalies flagged.")

    # --- Store results to S3 ---
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if s3_bucket not in existing:
        s3.create_bucket(Bucket=s3_bucket)
        print(f"Created bucket: {s3_bucket}")

    run_ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M")

    # Full scored results
    full_buf = io.StringIO()
    df.to_csv(full_buf, index=False)
    full_bytes = full_buf.getvalue().encode("utf-8")

    # Anomalies only
    anomalies = df[df["anomaly_count"] > 0].copy()
    alert_cols = [
        "timestamp", "avg_bytes", "total_bytes", "total_packets", "unique_flows",
        "bytes_per_flow", "bytes_per_packet", "src_dst_ratio",
        "iso_score", "zscore",
        "anomaly_iso", "anomaly_zscore", "anomaly_iqr",
        "anomaly_count", "anomaly_consensus", "severity",
    ]
    anomaly_buf = io.StringIO()
    anomalies[alert_cols].to_csv(anomaly_buf, index=False)
    anomaly_bytes = anomaly_buf.getvalue().encode("utf-8")
    anomaly_json  = json.dumps(
        anomalies[alert_cols].assign(
            timestamp=anomalies["timestamp"].astype(str)
        ).to_dict(orient="records"),
        indent=2,
    ).encode("utf-8")

    summary = {
        "run_timestamp":   run_ts,
        "generated_at":    datetime.utcnow().isoformat(),
        "n_buckets":       n_total,
        "n_iso":           n_iso,
        "n_zscore":        n_zscore,
        "n_iqr":           n_iqr,
        "n_consensus":     n_consensus,
        "n_critical":      int((df["severity"] == "critical").sum()),
        "time_range_start": str(df["timestamp"].min()),
        "time_range_end":   str(df["timestamp"].max()),
    }
    summary_bytes = json.dumps(summary, indent=2).encode("utf-8")

    for key_prefix in [f"anomalies/{run_ts}", "anomalies/latest"]:
        s3.put_object(Bucket=s3_bucket, Key=f"{key_prefix}/all_scored.csv",
                      Body=full_bytes, ContentType="text/csv")
        s3.put_object(Bucket=s3_bucket, Key=f"{key_prefix}/anomalies.csv",
                      Body=anomaly_bytes, ContentType="text/csv")
        s3.put_object(Bucket=s3_bucket, Key=f"{key_prefix}/anomalies.json",
                      Body=anomaly_json, ContentType="application/json")
        s3.put_object(Bucket=s3_bucket, Key=f"{key_prefix}/summary.json",
                      Body=summary_bytes, ContentType="application/json")

    iso_buf    = io.BytesIO(); joblib.dump(iso,    iso_buf);    iso_buf.seek(0)
    scaler_buf = io.BytesIO(); joblib.dump(scaler, scaler_buf); scaler_buf.seek(0)
    feat_bytes = json.dumps(feat_config).encode()

    for key_prefix in [f"models/{run_ts}", "models/latest"]:
        s3.put_object(Bucket=s3_bucket, Key=f"{key_prefix}/iso_forest.joblib",
                      Body=iso_buf.getvalue(), ContentType="application/octet-stream")
        s3.put_object(Bucket=s3_bucket, Key=f"{key_prefix}/scaler.joblib",
                      Body=scaler_buf.getvalue(), ContentType="application/octet-stream")
        s3.put_object(Bucket=s3_bucket, Key=f"{key_prefix}/features.json",
                      Body=feat_bytes, ContentType="application/json")

    print(f"Results → s3://{s3_bucket}/anomalies/{run_ts}/ and /anomalies/latest/")
    print(f"Model   → s3://{s3_bucket}/models/{run_ts}/ and /models/latest/")


@pipeline(
    name="packetbeat-traffic-anomaly-detection",
    description=(
        "15-min ES aggregation (flow.final:true) → feature engineering → "
        "Isolation Forest training → ensemble anomaly detection (IF + z-score + IQR) → S3"
    ),
)
def anomaly_detection_pipeline(
    es_host:           str   = "",
    es_user:           str   = "",
    es_password:       str   = "",
    es_index_prefix:   str   = "packetbeat-free5gc",
    fetch_days:        int   = 3,
    bucket_interval:   str   = "15m",
    # Isolation Forest params
    contamination:     float = 0.02,
    n_estimators:      int   = 200,
    # Ensemble params
    zscore_threshold:  float = 3.0,
    min_abs_deviation: float = 1000.0,
    iqr_multiplier:    float = 3.0,
    rolling_window:    int   = 24,
    # S3 config
    s3_endpoint:       str   = "",
    s3_access_key:     str   = "",
    s3_secret_key:     str   = "",
    s3_bucket:         str   = "traffic-anomalies",
    history_key:       str   = "history/aggregated_15m.csv",
):
    fetch_task = fetch_traffic_data(
        es_host=es_host,
        es_user=es_user,
        es_password=es_password,
        es_index_prefix=es_index_prefix,
        fetch_days=fetch_days,
        bucket_interval=bucket_interval,
    )
    fetch_task.set_caching_options(False)
    fetch_task.set_memory_limit("512Mi")
    fetch_task.set_memory_request("256Mi")
    fetch_task.set_cpu_limit("0.5")
    fetch_task.set_cpu_request("0.25")

    preprocess_task = preprocess_features(
        input_data=fetch_task.outputs["output_data"],
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_bucket=s3_bucket,
        history_key=history_key,
    )
    preprocess_task.set_caching_options(False)
    preprocess_task.set_memory_limit("1G")
    preprocess_task.set_memory_request("512Mi")
    preprocess_task.set_cpu_limit("1")

    train_task = train_anomaly_detector(
        input_data=preprocess_task.outputs["output_data"],
        contamination=contamination,
        n_estimators=n_estimators,
    )
    train_task.set_caching_options(False)
    train_task.set_memory_limit("2G")
    train_task.set_memory_request("1G")
    train_task.set_cpu_limit("2")

    detect_task = detect_anomalies_and_store(
        input_data=preprocess_task.outputs["output_data"],
        model_input=train_task.outputs["model_output"],
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_bucket=s3_bucket,
        zscore_threshold=zscore_threshold,
        min_abs_deviation=min_abs_deviation,
        iqr_multiplier=iqr_multiplier,
        rolling_window=rolling_window,
    )
    detect_task.set_caching_options(False)
    detect_task.set_memory_limit("1G")
    detect_task.set_memory_request("512Mi")
    detect_task.set_cpu_limit("1")


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=anomaly_detection_pipeline,
        package_path="anomaly_detection_pipeline.yaml",
    )
    print("Compiled → anomaly_detection_pipeline.yaml")