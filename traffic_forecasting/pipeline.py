import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics

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

    print(f"Forecasts saved → s3://{s3_bucket}/forecasts/{run_ts}/")
    print(f"Forecasts updated → s3://{s3_bucket}/forecasts/latest/")
    print(f"Model ({model_type}) saved → s3://{s3_bucket}/models/{run_ts}/")
    print(f"Model ({model_type}) updated → s3://{s3_bucket}/models/latest/")

    if model_type == "xgboost":
        with tempfile.TemporaryDirectory() as tmp:
            xgb_path = os.path.join(tmp, "model.bst")
            model.save_model(xgb_path)
            with open(xgb_path, "rb") as f:
                xgb_bytes = f.read()

        s3.put_object(Bucket=s3_bucket, Key=f"models/{run_ts}/model.bst",
                      Body=xgb_bytes, ContentType="application/octet-stream")
        s3.put_object(Bucket=s3_bucket, Key="models/latest/model.bst",
                      Body=xgb_bytes, ContentType="application/octet-stream")
        print(f"XGBoost native model saved → models/{run_ts}/model.bst + models/latest/model.bst ✓")