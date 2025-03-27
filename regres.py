from flask import Flask, request, jsonify
import json, base64, pickle
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

app = Flask(__name__)

def process_json_to_dataframe(data):
    """
    Process input JSON into a DataFrame with a datetime index and a numeric 'Value' column.
    Also returns the detected input type ("boolean" or "numeric").
    """
    ts_data = data["metricData"]["tsData"]
    records = []
    # Determine input type from the first record (assumes all are the same type)
    if "valueBoolean" in ts_data[0]:
        input_type = "boolean"
    elif "valueNumeric" in ts_data[0]:
        input_type = "numeric"
    else:
        input_type = "unknown"
    
    for entry in ts_data:
        time_ms = int(entry["timeMs"])
        dt = pd.to_datetime(time_ms, unit='ms')
        if "valueBoolean" in entry:
            value = int(entry["valueBoolean"])  # True->1, False->0
        elif "valueNumeric" in entry:
            value = float(entry["valueNumeric"])
        else:
            value = None
        records.append({"datetime": dt, "Value": value})
    
    df = pd.DataFrame(records)
    df.sort_values("datetime", inplace=True)
    df.set_index("datetime", inplace=True)
    df.dropna(inplace=True)
    return df, input_type

def fine_tune_model(df, seasonal_period=9):
    """
    Fine tune an Exponential Smoothing model using the provided DataFrame.
    """
    model = ExponentialSmoothing(
        df["Value"],
        trend='add',
        seasonal='add',
        seasonal_periods=seasonal_period
    ).fit()
    # Store additional attributes needed for forecasting
    model.last_timestamp = df.index[-1]
    if len(df.index) >= 2:
        model.avg_delta = (df.index[-1] - df.index[0]) / (len(df) - 1)
    else:
        model.avg_delta = pd.Timedelta(seconds=1)
    return model

def forecast_future(model, steps=9):
    """
    Forecast future values for the given number of steps.
    Use model.last_timestamp and model.avg_delta to generate future timestamps.
    """
    forecast = model.forecast(steps=steps)
    forecast_index = [model.last_timestamp + (i * model.avg_delta) for i in range(1, steps+1)]
    forecast.index = forecast_index
    return forecast

def forecast_to_json(forecast, input_type="numeric"):
    """
    Convert forecast (a Pandas Series with datetime index) to a JSON-friendly dict.
    Each timestamp is converted to milliseconds since epoch.
    If input_type is boolean, output field is "valueBoolean", else "valueNumeric".
    """
    ts_list = []
    for ts, value in forecast.items():
        time_ms = int(ts.timestamp() * 1000)
        if input_type == "boolean":
            out_value = True if value >= 0.5 else False
            ts_list.append({"timeMs": str(time_ms), "valueBoolean": out_value})
        else:
            ts_list.append({"timeMs": str(time_ms), "valueNumeric": value})
    return {"tsData": ts_list}

# ---------- /build_model endpoint ----------
@app.route('/build_model', methods=['POST'])
def build_model():
    try:
        data = request.get_json()
        df, input_type = process_json_to_dataframe(data)
        model = fine_tune_model(df, seasonal_period=9)
        model.input_type = input_type  # Attach input type for later forecasting
        
        # Use the exact metric name as provided in the JSON input
        metric_name = data["metricData"]["name"]
        
        # Serialize the model to a pickle object, then encode as base64 string
        model_pickle = pickle.dumps(model)
        model_b64 = base64.b64encode(model_pickle).decode('utf-8')
        
        # Return JSON response with the metric name and the serialized model string
        response = {
            "name": metric_name,
            "model": model_b64
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- /predict endpoint ----------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expect input JSON to include:
        # "model": base64 encoded model string,
        # "name": metric name,
        # "days": optional forecast horizon in days (default=1)
        data = request.get_json()
        model_b64 = data.get("model")
        if not model_b64:
            return jsonify({"error": "Missing model field"}), 400
        
        days = float(data.get("days", 1))
        steps = int(days * 9)  # 9 observations per day
        
        # Decode and load the model from the base64 string
        model_pickle = base64.b64decode(model_b64.encode('utf-8'))
        model = pickle.loads(model_pickle)
        
        # Generate forecast
        forecast = forecast_future(model, steps=steps)
        output_json = forecast_to_json(forecast, input_type=model.input_type)
        # Include the metric name in the output if provided
        output_json["name"] = data.get("name", "")
        
        return jsonify(output_json), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("starting server")
    app.run(debug=True,port=8088)
