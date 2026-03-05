import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics

@component(
    base_image="python:3.9",
    packages_to_install=["elasticsearch==8.12.0", "pandas"]
)
def fetch_hourly_aggregated(
    es_host:         str,
    es_user:         str,
    es_password:     str,
    es_index_prefix: str,
    fetch_days:      int,
    output_data:     Output[Dataset],
):
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
    for offset in range(fetch_days):
        day = today - timedelta(days=offset)
        idx = f"{es_index_prefix}-{day.strftime('%Y.%m.%d')}"
        if es.indices.exists(index=idx):
            indices.append(idx)
            print(f"{idx}")
        else:
            print(f"{idx} missing — skipping")

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
                                "lte": today.isoformat(),
                            }
                        }
                    }
                ],
                "filter": [
                    {
                        "range": {
                            "event.duration": {
                                "lte": 3600000000000
                            }
                        }
                    }
                ]
            }
        },
        "aggs": {
            "per_hour": {
                "date_histogram": {
                    "field":          "@timestamp",
                    "fixed_interval": "1h",
                    "min_doc_count":  1,
                },
                "aggs": {
                    "total_bytes":   {"sum":         {"field": "network.bytes"}},
                    "total_packets": {"sum":         {"field": "network.packets"}},
                    "unique_flows":  {"cardinality": {"field": "flow.id.keyword"}},
                    "avg_duration":  {"avg":         {"field": "event.duration"}},
                    "src_bytes":     {"sum":         {"field": "source.bytes"}},
                    "dst_bytes":     {"sum":         {"field": "destination.bytes"}},
                    "avg_bpp": {
                        "avg": {
                            "script": {
                                "source": (
                                    "doc['network.bytes'].size() > 0 && "
                                    "doc['network.packets'].size() > 0 "
                                    "? doc['network.bytes'].value / (doc['network.packets'].value + 1) "
                                    ": 0"
                                )
                            }
                        }
                    },
                }
            }
        }
    }

    resp    = es.search(index=",".join(indices), body=query)
    buckets = resp["aggregations"]["per_hour"]["buckets"]

    if not buckets:
        raise RuntimeError("No aggregation buckets returned — check index field names.")

    records = []
    for b in buckets:
        records.append({
            "timestamp":     b["key_as_string"],
            "total_bytes":   b["total_bytes"]["value"]   or 0,
            "total_packets": b["total_packets"]["value"] or 0,
            "unique_flows":  b["unique_flows"]["value"]  or 0,
            "avg_duration":  b["avg_duration"]["value"]  or 0,
            "src_bytes":     b["src_bytes"]["value"]     or 0,
            "dst_bytes":     b["dst_bytes"]["value"]     or 0,
            "bytes_per_pkt": b["avg_bpp"]["value"]       or 0,
        })

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(output_data.path, index=False)

    print(f"Fetched {len(df)} hourly buckets across {len(indices)} indices")
    print(f"Range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(df.tail(5).to_string(index=False))


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "boto3"]
)
def preprocess_and_save_history(
    input_data:  Input[Dataset],
    output_data: Output[Dataset],
    s3_endpoint:   str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_bucket:     str,
    history_key:   str,
):
    import pandas as pd
    import numpy as np
    import boto3, io
    from botocore.client import Config

    if not s3_endpoint:
        raise ValueError("s3_endpoint is required")
    if not s3_access_key:
        raise ValueError("s3_access_key is required")
    if not s3_secret_key:
        raise ValueError("s3_secret_key is required")

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    df = pd.read_csv(input_data.path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Hourly rows: {len(df)}")
    print(f"Range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    df["bytes_lag_1h"]   = df["total_bytes"].shift(1).fillna(df["total_bytes"])
    df["bytes_lag_2h"]   = df["total_bytes"].shift(2).fillna(df["total_bytes"])
    df["bytes_lag_3h"]   = df["total_bytes"].shift(3).fillna(df["total_bytes"])
    df["bytes_lag_6h"]   = df["total_bytes"].shift(6).fillna(df["total_bytes"])
    df["bytes_lag_12h"]  = df["total_bytes"].shift(12).fillna(df["total_bytes"])
    df["bytes_lag_24h"]  = df["total_bytes"].shift(24).fillna(df["total_bytes"])  # same hour yesterday
    df["bytes_lag_168h"] = df["total_bytes"].shift(168).fillna(df["total_bytes"]) # same hour last week

    df["bytes_rolling_3h"]  = df["total_bytes"].rolling(3,   min_periods=1).mean()
    df["bytes_rolling_6h"]  = df["total_bytes"].rolling(6,   min_periods=1).mean()
    df["bytes_rolling_12h"] = df["total_bytes"].rolling(12,  min_periods=1).mean()
    df["bytes_rolling_24h"] = df["total_bytes"].rolling(24,  min_periods=1).mean()
    df["bytes_rolling_7d"]  = df["total_bytes"].rolling(168, min_periods=1).mean()

    df["hour_of_day"]   = df["timestamp"].dt.hour
    df["day_of_week"]   = df["timestamp"].dt.dayofweek
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"]      = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"]       = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]       = np.cos(2 * np.pi * df["day_of_week"] / 7)

    try:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        s3.put_object(
            Bucket=s3_bucket,
            Key=history_key,
            Body=buf.getvalue().encode("utf-8"),
            ContentType="text/csv",
        )
        print(f"History saved → s3://{s3_bucket}/{history_key}")
    except Exception as e:
        print(f"Warning: could not save history to S3: {e}")

    df.to_csv(output_data.path, index=False)
    print(f"Training-ready rows: {len(df)}")


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "scikit-learn", "xgboost", "joblib"]
)
def train_model(
    input_data:       Input[Dataset],
    model_output:     Output[Model],
    metrics:          Output[Metrics],
    forecast_horizon: int,
):
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import xgboost as xgb
    import joblib, os, json

    df = pd.read_csv(input_data.path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    n_hours = len(df)
    print(f"Training on {n_hours} hourly rows ({n_hours/24:.1f} days)")

    feature_cols = [
        "total_packets", "unique_flows", "avg_duration",
        "src_bytes", "dst_bytes", "bytes_per_pkt",
        "bytes_lag_1h", "bytes_lag_2h", "bytes_lag_3h",
        "bytes_lag_6h", "bytes_lag_12h", "bytes_lag_24h", "bytes_lag_168h",
        "bytes_rolling_3h", "bytes_rolling_6h",
        "bytes_rolling_12h", "bytes_rolling_24h", "bytes_rolling_7d",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    ]
    target_col = "total_bytes"

    X = df[feature_cols].fillna(0)
    y = df[target_col]

    if n_hours < 168:
        print(f"Using Ridge regression ({n_hours} hours — below 168h XGBoost threshold)")
        model_type = "ridge"
        if n_hours > forecast_horizon:
            split = n_hours - forecast_horizon
            model = Ridge(alpha=1.0)
            model.fit(X.iloc[:split], y.iloc[:split])
            preds  = model.predict(X.iloc[split:])
            y_test = y.iloc[split:]
            mae    = mean_absolute_error(y_test, preds)
            rmse   = np.sqrt(mean_squared_error(y_test, preds))
            mape   = np.mean(np.abs((y_test.values - preds) / (y_test.values + 1))) * 100
            model.fit(X, y)
        else:
            model = Ridge(alpha=1.0)
            model.fit(X, y)
            preds = model.predict(X)
            mae   = mean_absolute_error(y, preds)
            rmse  = np.sqrt(mean_squared_error(y, preds))
            mape  = np.mean(np.abs((y.values - preds) / (y.values + 1))) * 100
    else:
        print(f"Using XGBoost ({n_hours} hours)")
        model_type = "xgboost"
        split = n_hours - forecast_horizon
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        mape  = np.mean(np.abs((y_test.values - preds) / (y_test.values + 1))) * 100

    print(f"MAE:  {mae:,.0f} bytes")
    print(f"RMSE: {rmse:,.0f} bytes")
    print(f"MAPE: {mape:.2f}%")

    metrics.log_metric("MAE",            float(mae))
    metrics.log_metric("RMSE",           float(rmse))
    metrics.log_metric("MAPE",           float(mape))
    metrics.log_metric("training_hours", float(n_hours))
    metrics.log_metric("model_type",     model_type)

    os.makedirs(model_output.path, exist_ok=True)
    joblib.dump(model, os.path.join(model_output.path, "model.joblib"))

    if model_type == "xgboost":
        booster = model.get_booster()
        booster.feature_names = None
        booster.feature_types = None
        booster.save_model(os.path.join(model_output.path, "model.json"))

    with open(os.path.join(model_output.path, "features.json"), "w") as f:
        json.dump(feature_cols, f)

    with open(os.path.join(model_output.path, "model_type.json"), "w") as f:
        json.dump({"type": model_type}, f)

    print(f"Model ({model_type}) + features + model_type saved ✓")


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "joblib", "xgboost", "scikit-learn", "boto3"]
)
def forecast_and_store_s3(
    input_data:       Input[Dataset],
    model_input:      Input[Model],
    s3_endpoint:      str,
    s3_access_key:    str,
    s3_secret_key:    str,
    s3_bucket:        str,
    forecast_horizon: int,
):
    import pandas as pd
    import numpy as np
    import joblib, os, json, tempfile
    from datetime import datetime, timedelta
    import boto3
    from botocore.client import Config
    import io

    if not s3_endpoint:
        raise ValueError("s3_endpoint is required")
    if not s3_access_key:
        raise ValueError("s3_access_key is required")
    if not s3_secret_key:
        raise ValueError("s3_secret_key is required")

    df = pd.read_csv(input_data.path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    model = joblib.load(os.path.join(model_input.path, "model.joblib"))

    with open(os.path.join(model_input.path, "features.json")) as f:
        feature_cols = json.load(f)

    with open(os.path.join(model_input.path, "model_type.json")) as f:
        model_type = json.load(f)["type"]

    print(f"Model type: {model_type}")

    last_row  = df.iloc[-1].copy()
    history   = df["total_bytes"].tolist()
    forecasts = []

    for i in range(1, forecast_horizon + 1):
        future_ts   = last_row["timestamp"] + timedelta(hours=i)
        hour_of_day = future_ts.hour
        day_of_week = future_ts.dayofweek

        row = {
            "total_packets":     last_row["total_packets"],
            "unique_flows":      last_row["unique_flows"],
            "avg_duration":      last_row["avg_duration"],
            "src_bytes":         last_row["src_bytes"],
            "dst_bytes":         last_row["dst_bytes"],
            "bytes_per_pkt":     last_row["bytes_per_pkt"],
            "bytes_lag_1h":      history[-1],
            "bytes_lag_2h":      history[-2]   if len(history) >= 2   else history[-1],
            "bytes_lag_3h":      history[-3]   if len(history) >= 3   else history[-1],
            "bytes_lag_6h":      history[-6]   if len(history) >= 6   else history[-1],
            "bytes_lag_12h":     history[-12]  if len(history) >= 12  else history[-1],
            "bytes_lag_24h":     history[-24]  if len(history) >= 24  else history[-1],
            "bytes_lag_168h":    history[-168] if len(history) >= 168 else history[-1],
            "bytes_rolling_3h":  np.mean(history[-3:]),
            "bytes_rolling_6h":  np.mean(history[-6:]),
            "bytes_rolling_12h": np.mean(history[-12:]),
            "bytes_rolling_24h": np.mean(history[-24:]),
            "bytes_rolling_7d":  np.mean(history[-168:]),
            "hour_sin":          np.sin(2 * np.pi * hour_of_day / 24),
            "hour_cos":          np.cos(2 * np.pi * hour_of_day / 24),
            "dow_sin":           np.sin(2 * np.pi * day_of_week / 7),
            "dow_cos":           np.cos(2 * np.pi * day_of_week / 7),
            "is_weekend":        int(day_of_week >= 5),
        }

        pred = float(model.predict(pd.DataFrame([row]))[0])
        pred = max(0, pred)

        forecasts.append({
            "timestamp":      future_ts.isoformat(),
            "forecast_bytes": pred,
            "forecast_mbps":  round(pred * 8 / 3600 / 1e6, 4),
            "generated_at":   datetime.utcnow().isoformat(),
            "training_hours": len(df),
            "horizon_hours":  forecast_horizon,
            "model_type":     model_type,
        })
        history.append(pred)

    forecast_df = pd.DataFrame(forecasts)
    print(forecast_df[["timestamp", "forecast_bytes", "forecast_mbps"]].to_string(index=False))

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

    run_ts     = datetime.utcnow().strftime("%Y-%m-%dT%H-%M")
    csv_buf    = io.StringIO()
    forecast_df.to_csv(csv_buf, index=False)
    csv_bytes  = csv_buf.getvalue().encode("utf-8")
    json_bytes = json.dumps(forecasts, indent=2).encode("utf-8")

    s3.put_object(Bucket=s3_bucket, Key=f"forecasts/{run_ts}/forecast.csv",
                  Body=csv_bytes, ContentType="text/csv")
    s3.put_object(Bucket=s3_bucket, Key=f"forecasts/{run_ts}/forecast.json",
                  Body=json_bytes, ContentType="application/json")
    s3.put_object(Bucket=s3_bucket, Key="forecasts/latest/forecast.csv",
                  Body=csv_bytes, ContentType="text/csv")
    s3.put_object(Bucket=s3_bucket, Key="forecasts/latest/forecast.json",
                  Body=json_bytes, ContentType="application/json")

    model_buf = io.BytesIO()
    joblib.dump(model, model_buf)
    model_buf.seek(0)
    model_bytes = model_buf.read()

    s3.put_object(Bucket=s3_bucket, Key=f"models/{run_ts}/model.joblib",
                  Body=model_bytes, ContentType="application/octet-stream")
    s3.put_object(Bucket=s3_bucket, Key=f"models/{run_ts}/features.json",
                  Body=json.dumps(feature_cols).encode(), ContentType="application/json")
    s3.put_object(Bucket=s3_bucket, Key=f"models/{run_ts}/model_type.json",
                  Body=json.dumps({"type": model_type}).encode(), ContentType="application/json")
    s3.put_object(Bucket=s3_bucket, Key="models/latest/model.joblib",
                  Body=model_bytes, ContentType="application/octet-stream")
    s3.put_object(Bucket=s3_bucket, Key="models/latest/features.json",
                  Body=json.dumps(feature_cols).encode(), ContentType="application/json")
    s3.put_object(Bucket=s3_bucket, Key="models/latest/model_type.json",
                  Body=json.dumps({"type": model_type}).encode(), ContentType="application/json")


    if model_type == "xgboost":
        with tempfile.TemporaryDirectory() as tmp:
            xgb_path = os.path.join(tmp, "model.json")
            booster = model.get_booster()
            booster.feature_names = None
            booster.feature_types = None
            model.save_model(xgb_path)
            with open(xgb_path, "rb") as f:
                xgb_bytes = f.read()

        s3.put_object(Bucket=s3_bucket, Key=f"models/{run_ts}/model.json",
                    Body=xgb_bytes, ContentType="application/json")
        s3.put_object(Bucket=s3_bucket, Key="models/latest/model.json",
                    Body=xgb_bytes, ContentType="application/json")
        print(f"XGBoost json model saved → models/{run_ts}/model.json + models/latest/model.json ✓")

@pipeline(
    name="packetbeat-traffic-forecasting-hourly",
    description="Hourly ES aggregation → feature engineering → train → forecast → S3",
)
def traffic_forecasting_pipeline(
    es_host:          str   = "",
    es_user:          str   = "",
    es_password:      str   = "",
    es_index_prefix:  str   = "packetbeat-free5gc",
    fetch_days:       int   = 9,
    forecast_horizon: int   = 168,   # default: forecast next 7 days in hours
    s3_endpoint:      str   = "",
    s3_access_key:    str   = "",
    s3_secret_key:    str   = "",
    s3_bucket:        str   = "traffic-forecasts",
    history_key:      str   = "history/hourly_aggregated.csv",
):
    fetch_task = fetch_hourly_aggregated(
        es_host=es_host,
        es_user=es_user,
        es_password=es_password,
        es_index_prefix=es_index_prefix,
        fetch_days=fetch_days,
    )
    fetch_task.set_caching_options(False)
    fetch_task.set_memory_limit("512Mi")
    fetch_task.set_memory_request("256Mi")
    fetch_task.set_cpu_limit("0.5")
    fetch_task.set_cpu_request("0.25")

    preprocess_task = preprocess_and_save_history(
        input_data=fetch_task.outputs["output_data"],
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_bucket=s3_bucket,
        history_key=history_key,
    )
    preprocess_task.set_memory_limit("1G")      # up from 512Mi
    preprocess_task.set_memory_request("512Mi")
    preprocess_task.set_cpu_limit("1")

    train_task = train_model(
        input_data=preprocess_task.outputs["output_data"],
        forecast_horizon=forecast_horizon,
    )
    train_task.set_memory_limit("4G")           # up from 2G
    train_task.set_memory_request("2G")
    train_task.set_cpu_limit("2")

    forecast_task = forecast_and_store_s3(
        input_data=preprocess_task.outputs["output_data"],
        model_input=train_task.outputs["model_output"],
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_bucket=s3_bucket,
        forecast_horizon=forecast_horizon,
    )
    forecast_task.set_memory_limit("1G")        # up from 512Mi
    forecast_task.set_memory_request("512Mi")
    forecast_task.set_cpu_limit("1")


from kfp import compiler

compiler.Compiler().compile(
    pipeline_func=traffic_forecasting_pipeline,
    package_path="traffic_forecasting_pipeline_hourly.yaml"
)
print("Compiled → traffic_forecasting_pipeline_hourly.yaml")