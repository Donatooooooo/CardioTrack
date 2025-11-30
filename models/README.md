# CardioTrack's Model Card
## Model Details
- Developed by: D. Boccuzzi, M. Capone, F. Rosmarino
- Model Date: November 11th, 2025
- Model Version: 4
- Model Type: RandomForestClassifier
### Training information
- Best hyperparameters tuned with a 5-fold cross validation:
    - `max_depth` 12
    - `n_estimators` 800
    - `class_weight` balanced
- Applied approaches:
During training, oversampling technique was applied to balance the dataset and reduce bias toward the majority class. This ensured that the model learned equally from positive and negative cases, improving prediction performance for the minority class.
- Training started at: 11:26:59 2025-11-12
- Training ended at: 11:34:29 2025-11-12

## Intended Use
### Primary intended uses
The CardioTrack ML system is designed to support early detection of heart failure by analyzing clinical features and identifying patients who may be at risk. Its purpose is to assist cardiologists in deciding when further diagnostic tests, monitoring, or preventive treatments are needed. The system is also intended for local public health authorities, who can use aggregated predictions to plan healthcare resources and implement prevention strategies within the population.
### Primary intended users
The primary users of the model are cardiologists and other qualified medical professionals who rely on clinical decision support tools. They are responsible for interpreting the model’s predictions in conjunction with the patient’s medical history and additional clinical information. Public health authorities may also use aggregated, non-individual results to support long-term planning and policy development.
### Out-of-scope use cases
The model should not be used without access to complete and reliable clinical features, and it is not suitable for real-time emergency triage or for predictive tasks not directly related to heart failure.

## Factors
### Relevant factors
Model performance may vary depending on patient characteristics that influence heart disease risk, as reflected in the contributions of individual clinical features. Age remains a relevant factor because it strongly correlates with cardiovascular conditions. In addition, features such as ST_Slope, ChestPainType, MaxHR, and ExerciseAngina have the largest impact on individual predictions, as highlighted by SHAP module for XAI. These features capture meaningful physiological and clinical differences among patients and explain why the model predicts higher or lower risk for specific individuals. Instrumentation and environmental factors are not relevant because the model operates on structured clinical data rather than on signals or images affected by measurement devices or environmental conditions.
### Evaluation Factors
The evaluation focuses on key clinical features that the model heavily relies on. The Relevant factors were chosen because they are both present in the dataset and have the largest impact on the model’s outputs, allowing clear interpretation of how predictions are made.

## Metrics
### Model Performance
- `F1 Score` 0.8990
- `Recall` 0.9019
- `Accuracy` 0.8876
- `ROC-AUC` 0.9399
### Variation approaches
The reported metrics were computed using the best model selected during cross validation for hyperparameter tuning, and evaluated on a completely independent test set. This setup was chosen because it provides a cleaner estimate of real-world performance, reduces the risk of overfitting to validation folds, and ensures that the results reflect the model’s generalization ability.

## Evaluation Data
### Datasets
The evaluation was performed using 276 of 918 (30%) observations of the Kaggle's [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction), which contains clinical data from both healthy individuals and patients diagnosed with heart failure.
### Motivation
This dataset was chosen because it provides a comprehensive set of relevant clinical features that capture key cardiovascular risk factors, enabling the model to perform early detection of heart failure in individual patients. Its publicly available nature ensures transparency.
### Preprocessing
Before evaluation, the data was preprocessed as follows:
- Normalization of continuous features using a Standard Scaler.
- One-hot encoding of categorical variables.
- Removal of the sex feature, as model performance was unchanged with or without it.

## Training Data
### Datasets
The training data mirrors the evaluation dataset, using 642 of 918 (70%) of the same Kaggle Heart Failure Prediction Dataset.
### Preprocessing
The training data underwent the same preprocessing steps as the evaluation data. Additionally, RandomOversampler technique was applied to balance the classes, ensuring that the model learned equally from positive (heart failure) and negative cases.

## Ethical Considerations
The Cardio Track ML system is intended to support clinical decision-making but not replace professional judgment. Ethical considerations include:
- **Privacy and security**: All patient data is processed on-premises, in accordance with hospital IT protocols, protecting sensitive health information.  
- **Transparency**: Feature importance with SHAP visualizations allow clinicians to interpret predictions.
- **Clinical responsibility**: Diagnosis must be combined with patient history, exams, and expert judgment. Misuse in isolation could lead to incorrect interventions.  

## Caveats and Recommendations
- **Inference time**: The model’s inference time is about 0.2 seconds, but it can varies with computing power where inference is run.
- **Limitations**: The model is trained on a specific public dataset and may not capture rare cardiovascular conditions or population-specific variations.
- **Data quality**: Accurate predictions depend on complete and correctly measured clinical features. Erroneous data can reduce performance. Missing data is not allowed.
- **Not for emergency triage**: Predictions are intended for early detection and planning, not for immediate emergency decision-making.
- **Periodic retraining**: To maintain accuracy, the model should be updated with newly collected clinical data to account for shifts in patient population or disease prevalence.  
