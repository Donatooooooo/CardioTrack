# **CARDIO TRACK - MACHINE LEARNING CANVAS**

**Designed for:** Giulio Mallardi  
**Designed by:** D. Boccuzzi, M. Capone, F. Rosmarino  
**Date:** 17/10/2025  
**Iteration:** 2  

---

## **1. Prediction Task**

Cardio Track ML system performs a **binary classification** task based on clinical data from individual patients, with the goal of predicting the presence or absence of heart disease.  
Specifically, the model analyzes each patient’s clinical features and risk factors to estimate the likelihood of developing heart failure.

There are two possible prediction outcomes:  
- **Positive:** when the patient shows indicators of heart failure.  
- **Negative:** when no signs of disease are detected.

---

## **2. Decisions**

The system’s predictions support **cardiologists** and **public health institutions (ASL)**.  
For positive cases, cardiologists can order further tests, start monitoring, and define personalized treatments.  
Aggregated results help public health institutions plan resources, prioritize facilities, and promote prevention and lifestyle improvements for long-term cardiovascular health.

---

## **3. Value Proposition**

The main end users are **cardiologists** and **local health authorities (ASL)**.  
For cardiologists, the system provides a reliable tool to assist in the early diagnosis of heart failure.  
For health authorities, it enables more efficient management of healthcare resources by optimizing the distribution of diagnostic and therapeutic services.

Overall, Cardio Track ML system aims to support **prevention** and **early detection** of heart failure, improving patient outcomes and reducing mortality rates.

---

## **4. Data Collection**

Data collection will be a **continuous and evolving process**.  
Real and high quality clinical data will be carefully labeled and verified by domain experts, ensuring data quality and consistency.  
New patient data collected through standardized clinical protocols will periodically update and improve the model, allowing it to adapt and learn over time.

---

## **5. Data Sources**

The ML system will rely on a **publicly available dataset** that includes clinical parameters from both healthy individuals and patients diagnosed with heart failure.  
The reference dataset is the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

---

## **6. Impact Simulation**

Before release, the model will undergo rigorous validation on an independent test set.

As a baseline, minimum thresholds are defined for the key evaluation metrics: **F1-score**, **recall** and **accuracy** ≥ 0.80 and **ROC-AUC** ≥ 0.85. These values ensure that the model maintains high discriminative capability while minimizing the risk of undetected clinical cases.

We will assess potential bias across demographic subgroups to ensure fairness and consistent model performance.
Any detected bias will be mitigated through rebalancing techniques or threshold adjustment to guarantee equitable treatment across all patient categories.

---

## **7. Making Predictions**

Predictions will be made on-demand, triggered whenever new or updated clinical data becomes available in the hospital database.  
Real-time processing is not required, but timely inference will support the decision-making workflow.  
All computations will be executed **on-premises**, using the existing hospital IT infrastructure to ensure **data privacy** and **security**.

---

## **8. Building Models**

Cardio Track ML system will use a **single main model** in production.  
Model updates will occur periodically as new data is integrated, or when a new version demonstrates statistically significant improvements in key metrics: **F1-score**, **recall**, **accuracy**, and **ROC-AUC**.

Model explainability will be ensured through the analysis of feature importance. Feature impact will be quantified by observing charts, allowing medical experts to interpret and validate the relevance of clinical factors used in the decision process.
---

## **9. Features**

The Heart Failure Prediction Dataset already provides a complete set of clinical features, so there is no need to extract them directly from medical exams or diagnostic reports.

**Included features:**  
Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, and ST_Slope.  

These features capture key cardiovascular risk factors such as hypertension, diabetes, hyperlipidemia, obesity, and other pre-existing heart conditions, making the dataset suitable for early heart failure diagnosis.

---

## **10. Monitoring**

After deployment, system performance will be continuously **monitored** to detect potential drifts or degradations over time.  
Key metrics include **F1-score**, **recall**, **accuracy**, and **ROC-AUC**, reviewed at regular intervals.  

Clinician feedback will also be collected to assess **usability** and **clinical relevance**, ensuring continuous model improvement and alignment with real-world medical needs.
