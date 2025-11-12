from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator


class HeartSample(BaseModel):
    Age: int
    Sex: Literal["M", "F"] | None = None
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"]
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: int
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float 
    ST_Slope: Literal["Up", "Flat", "Down"]

    @field_validator("Oldpeak")
    @classmethod
    def round_oldpeak(cls, v: float) -> float:
        return float(np.round(v, 2))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()])
