from __future__ import annotations

import pandas as pd
from predicting_outcomes_in_heart_failure.app.schema import HeartSample
import pytest


def test_heartsample_valid_creation():
    """HeartSample should be created correctly with valid input data."""
    sample = HeartSample(
        Age=50,
        ChestPainType="ATA",
        RestingBP=120,
        Cholesterol=220,
        FastingBS=0,
        RestingECG="Normal",
        MaxHR=160,
        ExerciseAngina="N",
        Oldpeak=1.23456,
        ST_Slope="Up",
    )

    assert sample.Age == 50
    assert sample.ChestPainType == "ATA"
    assert sample.RestingBP == 120
    assert sample.Cholesterol == 220


def test_heartsample_round_oldpeak():
    """Oldpeak should be rounded to 2 decimal places by the field validator."""
    sample = HeartSample(
        Age=50,
        ChestPainType="ASY",
        RestingBP=110,
        Cholesterol=200,
        FastingBS=1,
        RestingECG="ST",
        MaxHR=150,
        ExerciseAngina="Y",
        Oldpeak=1.2367,
        ST_Slope="Flat",
    )

    assert sample.Oldpeak == 1.24


def test_heartsample_invalid_literal():
    """Invalid literal values should raise a ValueError during validation."""
    with pytest.raises(ValueError):
        HeartSample(
            Age="Quaranta",
            ChestPainType="ATA",
            RestingBP=120,
            Cholesterol=200,
            FastingBS=0,
            RestingECG="Normal",
            MaxHR=150,
            ExerciseAngina="N",
            Oldpeak=0.5,
            ST_Slope="Up",
        )


def test_heartsample_to_dataframe():
    """to_dataframe() should return a single-row DataFrame with all fields."""
    sample = HeartSample(
        Age=60,
        ChestPainType="NAP",
        RestingBP=130,
        Cholesterol=210,
        FastingBS=1,
        RestingECG="LVH",
        MaxHR=140,
        ExerciseAngina="Y",
        Oldpeak=2.0,
        ST_Slope="Down",
    )

    df = sample.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, len(sample.model_dump()))
    assert df.loc[0, "Age"] == 60
    assert df.loc[0, "ST_Slope"] == "Down"
