# Developer Guide
This guide is for developers who want to modify the codebase, add new models, or extend the pipeline.

## Prerequisites

- **Python 3.11**
- **uv** - Fast Python package manager ([Official website](https://docs.astral.sh/uv/getting-started/installation/))
- **DVC** - Data Version Control ([Official website](https://dvc.org/))
- **Docker** and **Docker Compose** ([Official website](https://www.docker.com/))

## 1. Environment Setup

### Create and Update Virtual Environment

```bash
uv sync
```

This creates a `.venv` folder with Python 3.11. To activate it:

**Windows:**
```bash
.\.venv\Scripts\activate
```

**Unix/macOS:**
```bash
source ./.venv/bin/activate
```

To add packages:

```bash
uv add <your package>
```

## 2. Pull Data and Models (Optional)

If you want to use pre-trained models and processed data without running the full pipeline:

```bash
dvc remote modify origin --local access_key_id <your_dagshub_token>
dvc remote modify origin --local secret_access_key <your_dagshub_token>
dvc pull
```

This downloads all tracked data and model artifacts from the remote storage.

## 3. Run the ML Pipeline

### Pipeline Stages

The pipeline runs these stages in order:

| Stage | Description |
|-------|-------------|
| `download_data` | Downloads raw dataset from Kaggle |
| `preprocessing` | Cleans data, encodes features, scales numerical values |
| `split_data` | Splits into train/test sets (70/30) for each variant |
| `training` | Trains 3 models (Random Forest, Decision Tree, Logistic Regression) on 4 variants |
| `evaluation` | Evaluates models and saves metrics |

### Run Full Pipeline
```bash
dvc repro
```

### Run Specific Stages

```bash
# Run only preprocessing
dvc repro preprocessing

# Run training for the @champion model
dvc repro training@nosex-random_forest
```

## 4. Adding New Models

This section explains how to add a new ML model to the pipeline.

### Step 1: Update Configuration file

Edit `predicting_outcomes_in_heart_failure/config.py`:

```python
# Add model to valid models list
VALID_MODELS = ["logreg", "random_forest", "decision_tree", "your_new_model"]

# Add hyperparameter grid for your model
CONFIG_YOUR_MODEL = {
    "param1": [value1, value2],
    "param2": [value1, value2, value3],
}
```

### Step 2: Update Training Script

Edit `predicting_outcomes_in_heart_failure/modeling/train.py`:

1. Import the new config:
```python
from predicting_outcomes_in_heart_failure.config import (
    # ... existing imports
    CONFIG_YOUR_MODEL,
)
```

2. Add the model in `get_model_and_grid()`:
```python
def get_model_and_grid(model_name: str):
    # ... existing models ...

    elif model_name == "your_new_model":
        from package.module import YourModelClass

        estimator = YourModelClass(random_state=RANDOM_STATE)
        param_grid = CONFIG_YOUR_MODEL
        return estimator, param_grid
```

### Step 3: Update DVC Pipeline

Edit `dvc.yaml` to add entries for your new model:

```yaml
training:
  foreach:
    # ... existing entries ...
    - { variant: all,    model: your_new_model }
    - { variant: female, model: your_new_model }
    - { variant: male,   model: your_new_model }
    - { variant: nosex,  model: your_new_model }
  # ... rest unchanged ...

evaluation:
  foreach:
    # ... existing entries ...
    - { variant: all,    model: your_new_model }
    - { variant: female, model: your_new_model }
    - { variant: male,   model: your_new_model }
    - { variant: nosex,  model: your_new_model }
  # ... rest unchanged ...
```

### Step 4: Run the Pipeline

```bash
# Run only for your new model (all variants)
dvc repro training@all-your_new_model
dvc repro training@female-your_new_model
dvc repro training@male-your_new_model
dvc repro training@nosex-your_new_model

# Or run everything
dvc repro
```

### Output Files

After training, your model will generate:
- `models/{variant}/your_new_model.joblib` - Trained model
- `reports/{variant}/your_new_model/cv_results.csv` - Cross-validation results
- `reports/{variant}/your_new_model/cv_parameters.json` - Best parameters
- `metrics/test/{variant}/your_new_model.json` - Test metrics

---

## 5. Adding New Data Variants

To add a new data variant (e.g., age-filtered data):

### Step 1: Update Configuration file

Edit `predicting_outcomes_in_heart_failure/config.py`:

```python
VALID_VARIANTS = ["all", "female", "male", "nosex", "your_variant"]
```

### Step 2: Update Preprocessing

Edit `predicting_outcomes_in_heart_failure/data/preprocess.py` to generate the new variant CSV:

```python
# Add output for your variant
your_variant_df = df[your_filter_condition]
your_variant_df.to_csv(INTERIM_DATA_DIR / "preprocessed_your_variant.csv", index=False)
```

### Step 3: Update DVC Pipeline

Edit `dvc.yaml`:

```yaml
preprocessing:
  # ... existing ...
  outs:
    # ... existing outputs ...
    - data/interim/preprocessed_your_variant.csv

split_data:
  foreach: [all, female, male, nosex, your_variant]
  # ... rest unchanged ...

training:
  foreach:
    # ... add entries for your_variant ...
    - { variant: your_variant, model: logreg }
    - { variant: your_variant, model: random_forest }
    - { variant: your_variant, model: decision_tree }
  # ... rest unchanged ...

evaluation:
  foreach:
    # ... add entries for your_variant ...
    - { variant: your_variant, model: logreg }
    - { variant: your_variant, model: random_forest }
    - { variant: your_variant, model: decision_tree }
  # ... rest unchanged ...
```

## 6. Running Modified Code

After making changes to the codebase, follow this workflow:

### Quick Reference

| What Changed | Command |
|--------------|---------|
| Any stage script | `dvc repro` (runs affected stages only) |
| Specific stage | `dvc repro <stage_name>` |
| Force re-run | `dvc repro --force` |
| Single model/variant | `dvc repro training@{variant}-{model}` |

### Workflow

1. **Make your code changes** (models, preprocessing, etc.)

2. **Run the pipeline**:
   ```bash
   dvc repro
   ```
   DVC automatically detects which stages are affected by your changes and re-runs only those.

3. **Verify results**:
   ```bash
   # Check metrics
   cat metrics/test/{variant}/{model}.json

   # Check training reports
   cat reports/{variant}/{model}/cv_parameters.json
   ```
