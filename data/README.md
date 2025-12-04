# Dataset Card

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks](#supported-tasks)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
- [Dataset Creation](#dataset-creation)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Citation Information](#citation-information)



## Dataset Description

- **Homepage:** https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction  


### Dataset Summary

This dataset contains anonymized clinical data used to predict the risk of heart failure.  
It includes **918 patient records**, **11 clinical features**, and **one target variable**.  
The original dataset was downloaded from Kaggle and was created by merging five well-known cardiology datasets.

The version used in this project underwent additional preprocessing steps, including standardization, normalization, categorical encoding, and removal of the Sex feature. The resulting dataset is used for experimentation and model development.



### Supported Tasks

This dataset can be used for a variety of machine learning tasks, including:

- **Binary Classification**
 
  Predicting whether a patient has heart disease.  
- **Risk Scoring / Clinical Risk Stratification**   

   Estimating cardiac risk based on clinical variables.  
- **Explainable AI (XAI)**

   Useful for feature-importance analysis and interpretability.


### Languages

English **(en)**


## Dataset Structure

### Data Instances

Each instance represents one patient. Example:

| Age |Sex | ChestPainType | RestingBP | Cholesterol | FastingBS | RestingECG | MaxHR | ExerciseAngina | Oldpeak | ST_Slope | HeartDisease |
|-----|----|---------------|-----------|-------------|-----------|------------|-------|----------------|---------|----------|--------------|
| 54  | M  | ASY           | 140       | 239         | 0         | Normal     | 160   | N              | 1.2     | Flat     | 1            |



### Data Fields

| Field          | Type      | Description                                                   |
|----------------|-----------|---------------------------------------------------------------|
| Age            | int       | Patient age in years                                          |
| Sex            | binary    | Patient sex (M = male, F = female)                            |
| ChestPainType  | category  | Chest pain type (TA, ATA, NAP, ASY)                           |
| RestingBP      | int       | Resting blood pressure (mm Hg)                                |
| Cholesterol    | int       | Serum cholesterol (mg/dL)                                     |
| FastingBS      | binary    | Fasting blood sugar (1 if >120 mg/dL, 0 otherwise)            |
| RestingECG     | category  | Resting ECG results (Normal, ST, LVH)                         |
| MaxHR          | int       | Maximum heart rate achieved                                   |
| ExerciseAngina | binary    | Exercise-induced angina (Y/N)                                 |
| Oldpeak        | float     | ST depression relative to rest                                |
| ST_Slope       | category  | Slope of the ST segment (Up, Flat, Down)                      |
| HeartDisease   | binary    | Target variable (1 = disease, 0 = no disease)                 |



## Dataset Creation

### Source Data

The preprocessed dataset used in this project originates from the Kaggle dataset *“Heart Failure Prediction Dataset”*.  

The raw dataset was created by merging five widely-used cardiology datasets:

- Cleveland (303 samples)  
- Hungarian (294 samples)  
- Switzerland (123 samples)  
- Long Beach VA (200 samples)  
- Stalog (270 samples)

The Kaggle author selected the 11 common features and merged the datasets into a unified collection of **1,190 records**, then removed **272 duplicates**, resulting in **918 unique samples**.

All initial merging and normalization steps were performed by the dataset author on Kaggle.



### Annotations

No manual annotations were added.  
The target variable `HeartDisease` is already included in the original dataset.



### Personal and Sensitive Information

Although the dataset contains clinical information (sensitive under GDPR), it is fully anonymized:

- No personal identifiers (name, address, contact details, IDs).    
- All sources were already anonymized before publication.  
- No biometric or genetic data are included.

Thus, while clinically sensitive, the dataset does **not** pose identifiable privacy risks.



## Considerations for Using the Data

### Social Impact of Dataset

The dataset can support research and development of models for cardiac risk prediction and early detection.  

However:

- Models trained on this dataset **must not be used as standalone diagnostic tools**.  
- They should **not** be the sole basis for clinical decisions.  
- Misuse in healthcare contexts may lead to incorrect risk assessment.



### Discussion of Biases


This dataset may contain several sources of bias that can affect model performance and fairness:

- The data comes from multiple hospitals and countries, each with different patient profiles and clinical protocols. Some groups may be underrepresented.
- Source datasets used different diagnostic practices and measurement standards, which may introduce noise or inconsistency in labels and clinical values.
- Only 11 features are included, omitting other relevant clinical variables. This can cause proxy bias or oversimplification of cardiac risk.
- Some datasets are older and may not reflect current medical practices or population characteristics.



## Additional Information

### Dataset Curators

The original dataset was created and published by **[fedesoriano](https://www.kaggle.com/fedesoriano)** on Kaggle.  

The preprocessed dataset was curated by the **CardioTrack** team:

- [Fabrizio Rosmarino](https://github.com/Fabrizio250)  
- [Martina Capone](https://github.com/Martycap)  
- [Donato Boccuzzi](https://github.com/donatooooooo)

Work carried out as part of the *Software Engineering for AI-Enabled Systems* program at the University of Bari.

### Citation Information

If you use this datasets, please cite:

**Original Dataset**  
Soriano, F. (2021). *Heart Failure Prediction Dataset*. Kaggle.  
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

