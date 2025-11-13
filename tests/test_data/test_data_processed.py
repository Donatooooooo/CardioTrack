import great_expectations as gx
from gx_util import set_gx
import pandas as pd
from predicting_outcomes_in_heart_failure.config import (
    ASSET_NAME,
    PREPROCESSED_CSV,
    SOURCE_NAME,
    SUITE_NAME,
)


def run_test():
    suite.add_expectation(
        gx.expectations.ExpectTableColumnCountToEqual(value=len(expected_columns))
    )

    for col in expected_columns:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    for col in ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeOfType(column=col, type_="float")
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnMeanToBeBetween(column=col, min_value=-0.1, max_value=0.1)
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnStdevToBeBetween(column=col, min_value=0.9, max_value=1.1)
        )

    for col in ["Sex", "FastingBS", "ExerciseAngina", "HeartDisease"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(column=col, value_set=[0, 1])
        )

    for col in [
        "ChestPainType_ASY",
        "ChestPainType_ATA",
        "ChestPainType_NAP",
        "ChestPainType_TA",
        "RestingECG_LVH",
        "RestingECG_Normal",
        "RestingECG_ST",
        "ST_Slope_Down",
        "ST_Slope_Flat",
        "ST_Slope_Up",
    ]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(column=col, value_set=[True, False])
        )

    context.suites.add_or_update(suite)
    validation_definition = context.validation_definitions.add(
        gx.core.validation_definition.ValidationDefinition(
            name=ASSET_NAME + "_validation_processed",
            data=batch_definition,
            suite=suite,
        )
    )

    checkpoint = context.checkpoints.add(
        gx.checkpoint.checkpoint.Checkpoint(
            name=ASSET_NAME + "_checkpoint_validation_processed",
            validation_definitions=[validation_definition],
        )
    )

    checkpoint_result = checkpoint.run(batch_parameters={"dataframe": df})
    print(checkpoint_result.describe())


if __name__ == "__main__":
    df = pd.read_csv(PREPROCESSED_CSV)
    context, suite, batch_definition = set_gx(
        SOURCE_NAME + "_processed", ASSET_NAME + "_processed", SUITE_NAME + "_processed"
    )

    features_columns = [
        "Age",
        "Sex",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "MaxHR",
        "ExerciseAngina",
        "Oldpeak",
        "ChestPainType_ASY",
        "ChestPainType_ATA",
        "ChestPainType_NAP",
        "ChestPainType_TA",
        "RestingECG_LVH",
        "RestingECG_Normal",
        "RestingECG_ST",
        "ST_Slope_Down",
        "ST_Slope_Flat",
        "ST_Slope_Up",
    ]

    expected_columns = features_columns + ["HeartDisease"]
    run_test()
