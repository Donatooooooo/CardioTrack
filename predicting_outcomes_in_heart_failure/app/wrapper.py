import httpx
from loguru import logger
import pandas as pd

from predicting_outcomes_in_heart_failure.config import API_URL, FIGURES_DIR


async def _fetch_api(endpoint: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_URL}/{endpoint}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching {endpoint}: {e}")
            return {"error": str(e)}


class Wrapper:
    async def prediction_with_explanation(
        age,
        chest_pain_type,
        resting_bp,
        cholesterol,
        fasting_bs,
        resting_ecg,
        max_hr,
        exercise_angina,
        oldpeak,
        st_slope,
    ):
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
            "ST_Slope": st_slope,
        }

        async with httpx.AsyncClient() as client:
            try:
                pred_resp = await client.post(f"{API_URL}/predictions", json=payload)
                pred_resp.raise_for_status()
                pred_json = pred_resp.json()

                prediction_value = pred_json["data"]["prediction"]
                status = (
                    "🆘 At Risk (positive prediction)"
                    if prediction_value == 1
                    else "✅ No Risk Detected"
                )
                status_text = f"# Patient's status: {status}"
            except Exception as e:
                logger.error(f"Error making prediction: {e}")
                return f"Error during prediction: {str(e)}", ""

            try:
                expl_resp = await client.post(f"{API_URL}/explanations", json=payload)
                expl_resp.raise_for_status()
                expl_json = expl_resp.json()

                plot_rel_url = expl_json["data"].get("explanation_plot_url")
                if not plot_rel_url:
                    logger.warning("No explanation_plot_url found in /explanations response.")
                    return status_text, ""

                filename = plot_rel_url.split("/")[-1]
                plot_path = FIGURES_DIR / filename
                return status_text, str(plot_path)

            except Exception as e:
                logger.error(f"Error getting explanation: {e}")
                return status_text, ""

    async def batch_prediction(file):
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                df = pd.read_csv(file)

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
                        "ST_Slope": row["ST_Slope"],
                    }
                    payload.append(sample)

                response = await client.post(f"{API_URL}/batch-predictions", json=payload)
                response.raise_for_status()
                result = response.json()

                results = result["data"]["results"]
                df_results = pd.DataFrame(
                    [
                        {
                            "Patients's index": r["index"],
                            "Patient's status": "🆘 At Risk (positive prediction)"
                            if r["prediction"] == 1
                            else "✅ No Risk Detected",
                        }
                        for r in results
                    ]
                )

                return df_results
            except Exception as e:
                logger.error(f"Error making batch prediction: {e}")
                return pd.DataFrame({"error": [str(e)]})

    async def get_model_card():
        data = await _fetch_api("cards/model_card")

        card_lines = data.get("data").get("card_lines")
        return "\n".join(card_lines)

    async def get_dataset_card():
        data = await _fetch_api("cards/dataset_card")

        card_lines = data.get("data").get("card_lines")
        return "\n".join(card_lines)

    async def get_hyperparameters():
        data = await _fetch_api("model/hyperparameters")
        if "error" in data:
            return f"## Error\n{data['error']}"

        data = data.get("data", {}).get("hyperparameters", {}).get("cv", {})

        md = ""
        for key, value in data.items():
            md += f"- **{key}**: {value}\n"
        return md

    async def get_metrics():
        data = await _fetch_api("model/metrics")
        if "error" in data:
            return f"## Error\n{data['error']}"

        metrics = data.get("data", {}).get("metrics", {})
        if not metrics:
            return "## No metrics found"

        md = ""
        for key, value in metrics.items():
            md += f"- **{key}**: {value:.4f}\n"
        return md

    async def batch_explanation(file, patient_index: int):
        """Return SHAP plot (filepath) for a specific patient in the uploaded CSV."""
        try:
            df = pd.read_csv(file)
        except Exception as e:
            logger.error(f"Error reading CSV for batch explanation: {e}")
            return None

        try:
            idx = int(patient_index)
        except (TypeError, ValueError):
            logger.error(f"Invalid patient_index: {patient_index}")
            return None

        if idx < 0 or idx >= len(df):
            logger.error(f"patient_index {idx} out of range (0..{len(df) - 1})")
            return None

        row = df.iloc[idx]

        payload = {
            "Age": int(row["Age"]),
            "ChestPainType": row["ChestPainType"],
            "RestingBP": int(row["RestingBP"]),
            "Cholesterol": int(row["Cholesterol"]),
            "FastingBS": int(row["FastingBS"]),
            "RestingECG": row["RestingECG"],
            "MaxHR": int(row["MaxHR"]),
            "ExerciseAngina": row["ExerciseAngina"],
            "Oldpeak": round(float(row["Oldpeak"]), 2),
            "ST_Slope": row["ST_Slope"],
        }

        async with httpx.AsyncClient() as client:
            try:
                expl_resp = await client.post(f"{API_URL}/explanations", json=payload)
                expl_resp.raise_for_status()
                expl_json = expl_resp.json()

                plot_rel_url = expl_json["data"].get("explanation_plot_url")
                if not plot_rel_url:
                    logger.warning(
                        "No explanation_plot_url found in /explanations response (batch)."
                    )
                    return None

                filename = plot_rel_url.split("/")[-1]
                plot_path = FIGURES_DIR / filename

                return str(plot_path)

            except Exception as e:
                logger.error(f"Error getting batch explanation: {e}")
                return None
