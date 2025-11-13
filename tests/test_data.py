import great_expectations as gx
import pandas as pd
from predicting_outcomes_in_heart_failure.config import RAW_PATH

context = gx.get_context()
df = pd.read_csv(RAW_PATH)


data_source = context.data_sources.add_pandas(
    name="heart_data_source"
    )
data_asset = data_source.add_dataframe_asset(
    name="heart_failure"
    )

batch_definition = data_asset.add_batch_definition_whole_dataframe("batch_definition")
batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

suite_name = "heart_failure_data_quality" # Da mettere in config
suite = context.suites.add(
    gx.core.expectation_suite.ExpectationSuite(
        name=suite_name,
    )
)


features_columns = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]
expected_columns = features_columns + ["HeartDisease"]


suite.add_expectation(
    gx.expectations.ExpectTableColumnCountToEqual(value=len(expected_columns))
)

for col in expected_columns:
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

for col in ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]:
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(
            column=col, type_="float" if col == "Oldpeak" else "int"
        )
    )

for col in ["Sex", "ChestPainType", "RestingECG", "ST_Slope"]:
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(
            column=col, 
            type_="str"
        )
    )
    
for col in ["FastingBS", "HeartDisease"]:
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column=col,
            value_set=[0, 1]
        )
    )

suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeInSet(
        column="Sex",
        value_set=["M", "F"]
    )
)

suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeInSet(
        column="ExerciseAngina",
        value_set=["N", "Y"]
    )
)

suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeInSet(
        column="ST_Slope",
        value_set=["Flat", "Up", "Down"]
    )
)

suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeInSet(
        column="RestingECG",
        value_set=["Normal", "LVH", "ST"]
    )
)

suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeInSet(
        column="ChestPainType",
        value_set=["ASY", "NAP", "ATA", "TA"]
    )
)

suite.add_expectation(
    gx.expectations.ExpectCompoundColumnsToBeUnique(
        column_list=features_columns
    )
)

suite.add_expectation(
    gx.expectations.ExpectCompoundColumnsToBeUnique(
        column_list=expected_columns
    )
)

suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="Age", 
        min_value=18
    )
)

context.suites.add_or_update(suite)
validation_definition = context.validation_definitions.add(
    gx.core.validation_definition.ValidationDefinition(
        name="heart_failure_validation_definition",
        data=batch_definition,
        suite=suite,
    )
)

checkpoint = context.checkpoints.add(
    gx.checkpoint.checkpoint.Checkpoint(
        name="heart_failure_checkpoint",
        validation_definitions=[validation_definition],
    )
)

checkpoint_result = checkpoint.run(batch_parameters={"dataframe": df})
print(checkpoint_result.describe())