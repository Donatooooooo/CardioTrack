from loguru import logger
import httpx
import pandas as pd


async def fetch_api(endpoint: str):
    """Fetch data from API endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"http://localhost:8000{endpoint}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching {endpoint}: {e}")
            return {"error": str(e)}


async def predict_single(age, chest_pain_type, resting_bp, cholesterol,
                         fasting_bs, resting_ecg, max_hr, exercise_angina,
                         oldpeak, st_slope):
    async with httpx.AsyncClient() as client:
        try:
            payload = {
                "Age": age,
                "ChestPainType": chest_pain_type,
                "RestingBP": resting_bp,
                "Cholesterol": cholesterol,
                "FastingBS": fasting_bs,
                "RestingECG": resting_ecg,
                "MaxHR": max_hr,
                "ExerciseAngina": exercise_angina,
                "Oldpeak": round(oldpeak, 2),
                "ST_Slope": st_slope
            }
            response = await client.post(
                "http://localhost:8000/predictions", 
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Estrai la predizione dalla struttura di risposta
            prediction_value = result["data"]["prediction"]
            prediction = "Heart Disease" if prediction_value == 1 else "No Heart Disease"
            
            return f"**Prediction:** {prediction}\n\n**Prediction Value:** {prediction_value}"
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return f"Error: {str(e)}"


async def predict_batch(file):
    """Make batch prediction via API."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Leggi il file CSV
            df = pd.read_csv(file)
            
            # Converti il DataFrame in una lista di payload
            payload = []
            for _, row in df.iterrows():
                sample = {
                    "Age": int(row["Age"]),
                    "ChestPainType": row["ChestPainType"],
                    "RestingBP": int(row["RestingBP"]),
                    "Cholesterol": int(row["Cholesterol"]),
                    "FastingBS": int(row["FastingBS"]),
                    "RestingECG": row["RestingECG"],
                    "MaxHR": int(row["MaxHR"]),
                    "ExerciseAngina": row["ExerciseAngina"],
                    "Oldpeak": round(float(row["Oldpeak"]), 2),
                    "ST_Slope": row["ST_Slope"]
                }
                payload.append(sample)
            
            response = await client.post(
                "http://localhost:8000/batch-predictions",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Estrai i risultati dalla struttura di risposta
            results = result["data"]["results"]
            
            # Converti i risultati in DataFrame per visualizzazione
            df_results = pd.DataFrame([
                {
                    "Index": r["index"],
                    "Prediction": r["prediction"]
                }
                for r in results
            ])
            
            return df_results
        except Exception as e:
            logger.error(f"Error making batch prediction: {e}")
            return pd.DataFrame({"error": [str(e)]})

async def load_model_card():
    """Load model card from API."""
    data = await fetch_api('/cards/{Card_type}?card_type=model%20card')
    
    card_lines = data.get("data").get("card_lines")
    return "\n".join(card_lines)


async def load_dataset_card():
    """Load dataset card from API."""
    data = await fetch_api("/cards/{Card_type}?card_type=dataset%20card")
    
    card_lines = data.get("data").get("card_lines")
    return "\n".join(card_lines)



async def load_hyperparameters():
    data = await fetch_api("/model/hyperparameters")
    if "error" in data:
        return f"## Error\n{data['error']}"
    
    data = data.get("data", {}).get("hyperparameters", {}).get("cv", {})
    
    md = ""
    for key, value in data.items():
        md += f"- **{key}**: {value}\n"
    return md


async def load_metrics():
    data = await fetch_api("/model/metrics")
    if "error" in data:
        return f"## Error\n{data['error']}"

    metrics = data.get("data", {}).get("metrics", {})
    if not metrics:
        return "## No metrics found"

    md = ""
    for key, value in metrics.items():
        md += f"- **{key}**: {value:.4f}\n"
    return md