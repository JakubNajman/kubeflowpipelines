import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics


@component(
    base_image="python:3.9",
    packages_to_install=["elasticsearch==8.12.0", "pandas"]
)
def fetch_last_n_days_from_elasticsearch(
    es_host:         str,
    es_user:         str,
    es_password:     str,
    es_index_prefix: str,
    fetch_days:      int,
    output_data:     Output[Dataset],
):
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import scan
    import pandas as pd
    from datetime import datetime, timedelta

    if not es_host:
        raise ValueError("es_host is required — fill it in the KFP UI before running")
    if not es_user:
        raise ValueError("es_user is required — fill it in the KFP UI before running")
    if not es_password:
        raise ValueError("es_password is required — fill it in the KFP UI before running")

    es = Elasticsearch(
        es_host,
        basic_auth=(es_user, es_password),
        verify_certs=False,
    )

    today = datetime.utcnow().date()

    indices_to_fetch = []
    for offset in range(fetch_days):
        day = today - timedelta(days=offset)
        idx = f"{es_index_prefix}-{day.strftime('%Y.%m.%d')}"
        if es.indices.exists(index=idx):
            indices_to_fetch.append(idx)
            print(f"Found: {idx}")
        else:
            print(f"Missing: {idx} — skipping")

    if not indices_to_fetch:
        raise RuntimeError(
            f"No indices found for the last {fetch_days} days "
            f"with prefix '{es_index_prefix}'."
        )

    print(f"\nFetching {len(indices_to_fetch)} indices: {indices_to_fetch}")

    query = {
        "query": {"match_all": {}},
        "_source": [
            "@timestamp",
            "network.bytes",
            "network.packets",
            "flow.id",
            "network.protocol",
            "network.direction",
            "source.bytes",
            "destination.bytes",
            "event.duration",
        ]
    }

    records = []
    for idx in indices_to_fetch:
        day_records = []
        for hit in scan(es, index=idx, query=query, size=5000):
            day_records.append(hit["_source"])
        print(f"  {idx}: {len(day_records):,} records")
        records.extend(day_records)

    df = pd.DataFrame(records)
    df["_fetched_date"] = str(today)
    df.to_csv(output_data.path, index=False)
    print(f"\nTotal fetched: {len(df):,} records across {len(indices_to_fetch)} days")

@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "boto3"]
)
def preprocess_and_merge(
    input_data:    Input[Dataset],
    output_data:   Output[Dataset],
    s3_endpoint:   str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_bucket:     str,
    history_key:   str,
):
    import pandas as pd
    import numpy as np
    import boto3
    from botocore.client import Config
    import io

    if not s3_endpoint:
        raise ValueError("s3_endpoint is required — fill it in the KFP UI before running")
    if not s3_access_key:
        raise ValueError("s3_access_key is required — fill it in the KFP UI before running")
    if not s3_secret_key:
        raise ValueError("s3_secret_key is required — fill it in the KFP UI before running")

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    df = pd.read_csv(input_data.path)

    df["@timestamp"] = pd.to_datetime(
        df["@timestamp"],
        format="%b %d, %Y @ %H:%M:%S.%f",
        errors="coerce"
    )
    mask = df["@timestamp"].isna()
    if mask.any():
        df.loc[mask, "@timestamp"] = pd.to_datetime(
            df.loc[mask, "@timestamp"], utc=True, errors="coerce"
        ).dt.tz_localize(None)

    df = df.dropna(subset=["@timestamp"])
    if df["@timestamp"].dt.tz is not None:
        df["@timestamp"] = df["@timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

    for col in ["network.bytes", "network.packets", "source.bytes",
                "destination.bytes", "event.duration"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["bytes_per_packet"] = df["network.bytes"] / (df["network.packets"] + 1)

    daily_df = (
        df.set_index("@timestamp")
        .resample("1D")
        .agg(
            total_bytes  =("network.bytes",     "sum"),
            total_packets=("network.packets",   "sum"),
            unique_flows =("flow.id",           "nunique"),
            avg_duration =("event.duration",    "mean"),
            src_bytes    =("source.bytes",      "sum"),
            dst_bytes    =("destination.bytes", "sum"),
            bytes_per_pkt=("bytes_per_packet",  "mean"),
        )
        .reset_index()
        .rename(columns={"@timestamp": "date"})
    )

    daily_df = daily_df[daily_df["total_bytes"] > 0].reset_index(drop=True)

    print(f"Aggregated into {len(daily_df)} daily rows:")
    print(daily_df[["date", "total_bytes", "total_packets", "unique_flows"]].to_string(index=False))

    try:
        buf = io.StringIO()
        daily_df.to_csv(buf, index=False)
        s3.put_object(
            Bucket=s3_bucket,
            Key=history_key,
            Body=buf.getvalue().encode("utf-8"),
            ContentType="text/csv",
        )
        print(f"History saved → s3://{s3_bucket}/{history_key}")
    except Exception as e:
        print(f"Warning: could not save history to S3: {e}")

    daily_df["bytes_lag_1"]      = daily_df["total_bytes"].shift(1).fillna(daily_df["total_bytes"])
    daily_df["bytes_lag_2"]      = daily_df["total_bytes"].shift(2).fillna(daily_df["total_bytes"])
    daily_df["bytes_lag_3"]      = daily_df["total_bytes"].shift(3).fillna(daily_df["total_bytes"])
    daily_df["bytes_lag_7"]      = daily_df["total_bytes"].shift(7).fillna(daily_df["total_bytes"])
    daily_df["bytes_rolling_7d"] = daily_df["total_bytes"].expanding(min_periods=1).mean()
    daily_df["bytes_rolling_3d"] = daily_df["total_bytes"].rolling(3, min_periods=1).mean()

    daily_df.to_csv(output_data.path, index=False)
    print(f"Training-ready rows: {len(daily_df)}")


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
    import joblib, os

    df = pd.read_csv(input_data.path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    n_days = len(df)
    print(f"Training on {n_days} day(s) of history")

    feature_cols = [
        "total_packets", "unique_flows", "avg_duration",
        "src_bytes", "dst_bytes", "bytes_per_pkt",
        "bytes_lag_1", "bytes_lag_2", "bytes_lag_3", "bytes_lag_7",
        "bytes_rolling_7d", "bytes_rolling_3d",
    ]
    target_col = "total_bytes"

    X = df[feature_cols].fillna(0)
    y = df[target_col]

    if n_days < 21:
        print(f"Using Ridge regression ({n_days} days — below 21-day XGBoost threshold)")
        if n_days > forecast_horizon:
            split = n_days - forecast_horizon
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae   = mean_absolute_error(y_test, preds)
            rmse  = np.sqrt(mean_squared_error(y_test, preds))
            mape  = np.mean(np.abs((y_test.values - preds) / (y_test.values + 1))) * 100
            model.fit(X, y)
        else:
            model = Ridge(alpha=1.0)
            model.fit(X, y)
            preds = model.predict(X)
            mae   = mean_absolute_error(y, preds)
            rmse  = np.sqrt(mean_squared_error(y, preds))
            mape  = np.mean(np.abs((y.values - preds) / (y.values + 1))) * 100
    else:
        print("Using XGBoost")
        split = n_days - forecast_horizon
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        model = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, subsample=0.8,
            random_state=42, verbosity=0,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        mape  = np.mean(np.abs((y_test.values - preds) / (y_test.values + 1))) * 100

    print(f"MAE:  {mae:,.0f} bytes")
    print(f"RMSE: {rmse:,.0f} bytes")
    print(f"MAPE: {mape:.2f}%")

    metrics.log_metric("MAE",           float(mae))
    metrics.log_metric("RMSE",          float(rmse))
    metrics.log_metric("MAPE",          float(mape))
    metrics.log_metric("training_days", float(n_days))

    os.makedirs(model_output.path, exist_ok=True)
    joblib.dump(model, os.path.join(model_output.path, "model.joblib"))
    print("Model saved ✓")


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "joblib", "xgboost", "boto3"]
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
    import joblib, os, json
    from datetime import datetime, timedelta
    import boto3
    from botocore.client import Config
    import io

    if not s3_endpoint:
        raise ValueError("s3_endpoint is required — fill it in the KFP UI before running")
    if not s3_access_key:
        raise ValueError("s3_access_key is required — fill it in the KFP UI before running")
    if not s3_secret_key:
        raise ValueError("s3_secret_key is required — fill it in the KFP UI before running")

    df = pd.read_csv(input_data.path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    model = joblib.load(os.path.join(model_input.path, "model.joblib"))

    feature_cols = [
        "total_packets", "unique_flows", "avg_duration",
        "src_bytes", "dst_bytes", "bytes_per_pkt",
        "bytes_lag_1", "bytes_lag_2", "bytes_lag_3", "bytes_lag_7",
        "bytes_rolling_7d", "bytes_rolling_3d",
    ]

    last_row  = df.iloc[-1].copy()
    forecasts = []
    history   = df["total_bytes"].tolist()

    for i in range(1, forecast_horizon + 1):
        future_date = last_row["date"] + timedelta(days=i)
        row = {
            "total_packets":    last_row["total_packets"],
            "unique_flows":     last_row["unique_flows"],
            "avg_duration":     last_row["avg_duration"],
            "src_bytes":        last_row["src_bytes"],
            "dst_bytes":        last_row["dst_bytes"],
            "bytes_per_pkt":    last_row["bytes_per_pkt"],
            "bytes_lag_1":      history[-1],
            "bytes_lag_2":      history[-2] if len(history) >= 2 else history[-1],
            "bytes_lag_3":      history[-3] if len(history) >= 3 else history[-1],
            "bytes_lag_7":      history[-7] if len(history) >= 7 else history[-1],
            "bytes_rolling_7d": np.mean(history[-7:]),
            "bytes_rolling_3d": np.mean(history[-3:]),
        }
        pred = float(model.predict(pd.DataFrame([row]))[0])
        forecasts.append({
            "date":           future_date.isoformat(),
            "forecast_bytes": pred,
            "generated_at":   datetime.utcnow().isoformat(),
            "training_days":  len(df),
            "horizon_days":   forecast_horizon,
        })
        history.append(pred)

    forecast_df = pd.DataFrame(forecasts)
    print(forecast_df.to_string(index=False))

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

    run_date   = datetime.utcnow().strftime("%Y-%m-%d")
    csv_buf    = io.StringIO()
    forecast_df.to_csv(csv_buf, index=False)
    csv_bytes  = csv_buf.getvalue().encode("utf-8")
    json_bytes = json.dumps(forecasts, indent=2).encode("utf-8")

    s3.put_object(Bucket=s3_bucket, Key=f"forecasts/{run_date}/forecast.csv",
                  Body=csv_bytes, ContentType="text/csv")
    s3.put_object(Bucket=s3_bucket, Key=f"forecasts/{run_date}/forecast.json",
                  Body=json_bytes, ContentType="application/json")
    s3.put_object(Bucket=s3_bucket, Key="forecasts/latest/forecast.csv",
                  Body=csv_bytes, ContentType="text/csv")
    s3.put_object(Bucket=s3_bucket, Key="forecasts/latest/forecast.json",
                  Body=json_bytes, ContentType="application/json")

    model_buf = io.BytesIO()
    joblib.dump(model, model_buf)
    model_buf.seek(0)
    s3.put_object(Bucket=s3_bucket, Key=f"models/{run_date}/model.joblib",
                  Body=model_buf.read(), ContentType="application/octet-stream")
    print(f"Saved → s3://{s3_bucket}/forecasts/{run_date}/ and models/{run_date}/")
    print("Updated forecasts/latest/ ✓")



@pipeline(
    name="packetbeat-traffic-forecasting",
    description="Daily: fetch last N days from ES → aggregate → train → forecast → store S3",
)
def traffic_forecasting_pipeline(
    es_host:          str = "",
    es_user:          str = "",
    es_password:      str = "",
    es_index_prefix:  str = "packetbeat-free5gc",
    fetch_days:       int = 9,
    forecast_horizon: int = 7,
    s3_endpoint:      str = "",
    s3_access_key:    str = "",
    s3_secret_key:    str = "",
    s3_bucket:        str = "traffic-forecasts",
    history_key:      str = "history/daily_aggregated.csv",
):
    fetch_task = fetch_last_n_days_from_elasticsearch(
        es_host=es_host,
        es_user=es_user,
        es_password=es_password,
        es_index_prefix=es_index_prefix,
        fetch_days=fetch_days,
    )

    preprocess_task = preprocess_and_merge(
        input_data=fetch_task.outputs["output_data"],
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_bucket=s3_bucket,
        history_key=history_key,
    )

    train_task = train_model(
        input_data=preprocess_task.outputs["output_data"],
        forecast_horizon=forecast_horizon,
    )
    train_task.set_memory_limit("2G")
    train_task.set_cpu_limit("2")

    forecast_and_store_s3(
        input_data=preprocess_task.outputs["output_data"],
        model_input=train_task.outputs["model_output"],
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_bucket=s3_bucket,
        forecast_horizon=forecast_horizon,
    )


from kfp import compiler

compiler.Compiler().compile(
    pipeline_func=traffic_forecasting_pipeline,
    package_path="traffic_forecasting_pipeline.yaml"
)
print("Compiled → traffic_forecasting_pipeline.yaml")