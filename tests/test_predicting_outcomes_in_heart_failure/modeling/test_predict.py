import pandas as pd

from predicting_outcomes_in_heart_failure.config import INPUT_COLUMNS
from predicting_outcomes_in_heart_failure.modeling.predict import preprocessing


def test_preprocessing_produces_expected_columns(sample_raw_df_single):
    """

    Given as input a *raw* sample with all the original columns
    we expect that after preprocessing:

    * does not raise any error
    * returns a DataFrame
    * with columns exactly equal to `INPUT_COLUMNS` (same order)
    * with no NaN values
    """


    processed = preprocessing(sample_raw_df_single)

    assert isinstance(processed, pd.DataFrame)

    assert list(processed.columns) == INPUT_COLUMNS

    assert not processed.isna().any().any()



def test_preprocessing_maps_sex_correctly(sample_raw_df_two_rows):
    """
    Given a DataFrame with Sex = [M, F],
    we expect that after the preprocessing:
      - Sex is mapped to [1, 0]
    """


    processed = preprocessing(sample_raw_df_two_rows)


    sex_values = processed["Sex"].tolist()
    assert sex_values == [1, 0]


def test_preprocessing_encodes_exerciseAngina_binary(sample_raw_df_two_rows):
    """
    Given ExerciseAngina = [Y, N],
    we expect that after preprocessing:

      - ExerciseAngina is mapped to [1, 0] (Y -> 1, N -> 0)
    """
    processed = preprocessing(sample_raw_df_two_rows)

    ea_values = processed["ExerciseAngina"].tolist()
    assert ea_values == [1, 0]

def test_preprocessing_one_hot_chestPainType(sample_raw_df_two_rows):
    """
    Given ChestPainType = [ASY, NAP],
    we expect one-hot encoded columns for ChestPainType
    to be present and consistent with the raw input.
    """
    processed = preprocessing(sample_raw_df_two_rows)

    assert "ChestPainType_ASY" in processed.columns
    assert "ChestPainType_NAP" in processed.columns
    assert "ChestPainType_TA" in processed.columns

    # Row 0: ASY -> ASY=1, NAP=0, TA=0
    assert processed.loc[0, "ChestPainType_ASY"] == 1
    assert processed.loc[0, "ChestPainType_NAP"] == 0
    assert processed.loc[0, "ChestPainType_TA"] == 0

    # Row 1: NAP -> ASY=0, NAP=1, TA=0
    assert processed.loc[1, "ChestPainType_ASY"] == 0
    assert processed.loc[1, "ChestPainType_NAP"] == 1
    assert processed.loc[1, "ChestPainType_TA"] == 0

def test_preprocessing_one_hot_restingecg(sample_raw_df_two_rows):
    """
    Given RestingECG = [Normal, ST],
    we expect one-hot encoded columns for RestingECG
    to be present and consistent with the raw input.
    """
    processed = preprocessing(sample_raw_df_two_rows)

    assert "RestingECG_Normal" in processed.columns
    assert "RestingECG_ST" in processed.columns

    # Row 0: Normal
    assert processed.loc[0, "RestingECG_Normal"] == 1
    assert processed.loc[0, "RestingECG_ST"] == 0

    # Row 1: ST
    assert processed.loc[1, "RestingECG_Normal"] == 0
    assert processed.loc[1, "RestingECG_ST"] == 1

def test_preprocessing_one_hot_st_slope(sample_raw_df_two_rows):
    """
    Given ST_Slope = [Up, Flat],
    we expect one-hot encoded columns for ST_Slope
    to be present and consistent with the raw input.
    """
    processed = preprocessing(sample_raw_df_two_rows)

    assert "ST_Slope_Up" in processed.columns
    assert "ST_Slope_Flat" in processed.columns
    assert "ST_Slope_Down" in processed.columns

    # Row 0: Up
    assert processed.loc[0, "ST_Slope_Up"] == 1
    assert processed.loc[0, "ST_Slope_Flat"] == 0
    assert processed.loc[0, "ST_Slope_Down"] == 0

    # Row 1: Flat
    assert processed.loc[1, "ST_Slope_Up"] == 0
    assert processed.loc[1, "ST_Slope_Flat"] == 1
    assert processed.loc[1, "ST_Slope_Down"] == 0


def test_preprocessing_adds_missing_dummy_columns_with_zeros():
    """
    Given input where some categorical levels never appear
    (e.g., no TA for ChestPainType, no Flat/Down for ST_Slope),
    we still expect the corresponding one-hot columns to exist
    in the preprocessed DataFrame, filled with zeros.
    """

    # All rows use only a subset of possible categories
    sample_df = pd.DataFrame(
        {
            "Age": [50, 60, 55],
            "Sex": ["M", "F", "M"],
            "ChestPainType": ["ASY", "ASY", "ASY"],  # only ASY, never NAP/TA/etc.
            "RestingBP": [130, 140, 135],
            "Cholesterol": [220, 250, 230],
            "FastingBS": [0, 1, 0],
            "RestingECG": ["Normal", "Normal", "Normal"],  # only Normal
            "MaxHR": [150, 140, 145],
            "ExerciseAngina": ["Y", "N", "Y"],
            "Oldpeak": [1.0, 0.0, 0.5],
            "ST_Slope": ["Up", "Up", "Up"],  # only Up, never Flat/Down
        }
    )

    processed = preprocessing(sample_df)

    # ChestPainType: only ASY in input, but we still expect other dummies to exist
    assert "ChestPainType_ASY" in processed.columns
    assert "ChestPainType_NAP" in processed.columns
    assert "ChestPainType_TA" in processed.columns

    # Since NAP and TA never appeared, their columns should be all zeros
    assert (processed["ChestPainType_NAP"] == 0).all()
    assert (processed["ChestPainType_TA"] == 0).all()

    # RestingECG: only Normal in input, but ST/LVH columns should still exist and be 0
    assert "RestingECG_Normal" in processed.columns
    assert "RestingECG_ST" in processed.columns
    assert "RestingECG_LVH" in processed.columns

    assert (processed["RestingECG_ST"] == 0).all()
    assert (processed["RestingECG_LVH"] == 0).all()

    # ST_Slope: only Up in input, but Flat and Down should exist and be 0
    assert "ST_Slope_Up" in processed.columns
    assert "ST_Slope_Flat" in processed.columns
    assert "ST_Slope_Down" in processed.columns

    assert (processed["ST_Slope_Flat"] == 0).all()
    assert (processed["ST_Slope_Down"] == 0).all()
