import joblib
import numpy as np
import pandas as pd
from predicting_outcomes_in_heart_failure.data.preprocess import (
    generate_gender_splits,
    preprocessing,
    save_scaler_artifact,
)
import pytest
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def mock_paths(monkeypatch, tmp_path):
    import predicting_outcomes_in_heart_failure.data.preprocess as preprocess_module

    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(preprocess_module, "PREPROCESS_ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(preprocess_module, "SCALER_PATH", artifacts_dir / "scaler.pkl")
    monkeypatch.setattr(preprocess_module, "FEMALE_CSV", tmp_path / "female.csv")
    monkeypatch.setattr(preprocess_module, "MALE_CSV", tmp_path / "male.csv")
    monkeypatch.setattr(preprocess_module, "NOSEX_CSV", tmp_path / "nosex.csv")
    monkeypatch.setattr(preprocess_module, "RAW_PATH", tmp_path / "raw" / "heart.csv")
    monkeypatch.setattr(preprocess_module, "PREPROCESSED_CSV", tmp_path / "preprocessed.csv")
    monkeypatch.setattr(preprocess_module, "TARGET_COL", "HeartDisease")
    monkeypatch.setattr(
        preprocess_module,
        "NUM_COLS_DEFAULT",
        ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"],
    )

    return tmp_path


class TestSaveScalerArtifact:
    def test_save_scaler_creates_directory(self, mock_paths):
        """
        Test that the function creates the artifacts directory if it doesn't exist.
        """

        scaler = StandardScaler()
        scaler.fit([[0, 0], [1, 1]])

        save_scaler_artifact(scaler)

        assert (mock_paths / "artifacts").exists()
        assert (mock_paths / "artifacts" / "scaler.pkl").exists()

    def test_save_scaler_saves_correctly(self, mock_paths):
        """
        Test that the scaler is saved and can be loaded back.
        """

        scaler = StandardScaler()
        test_data = [[0, 0], [1, 1], [2, 2]]
        scaler.fit(test_data)

        save_scaler_artifact(scaler)

        loaded_scaler = joblib.load(mock_paths / "artifacts" / "scaler.pkl")
        assert isinstance(loaded_scaler, StandardScaler)
        np.testing.assert_array_almost_equal(loaded_scaler.mean_, scaler.mean_)
        np.testing.assert_array_almost_equal(loaded_scaler.scale_, scaler.scale_)

    def test_save_scaler_overwrites_existing(self, mock_paths):
        """
        Test that saving a scaler overwrites an existing one.
        """

        scaler1 = StandardScaler()
        scaler1.fit([[0, 0], [1, 1]])
        save_scaler_artifact(scaler1)

        scaler2 = StandardScaler()
        scaler2.fit([[10, 10], [20, 20]])
        save_scaler_artifact(scaler2)

        loaded_scaler = joblib.load(mock_paths / "artifacts" / "scaler.pkl")
        np.testing.assert_array_almost_equal(loaded_scaler.mean_, scaler2.mean_)

    def test_save_unfitted_scaler(self, mock_paths):
        """
        Test saving an unfitted scaler.
        """

        scaler = StandardScaler()
        save_scaler_artifact(scaler)

        loaded_scaler = joblib.load(mock_paths / "artifacts" / "scaler.pkl")
        assert isinstance(loaded_scaler, StandardScaler)
        assert not hasattr(loaded_scaler, "mean_")


class TestGenerateGenderSplits:
    @pytest.fixture
    def sample_df_with_sex(self):
        """
        Balanced dataset including Sex column.
        """

        return pd.DataFrame(
            {
                "Age": [45, 50, 55, 60, 65],
                "Sex": [0, 1, 0, 1, 0],
                "Cholesterol": [200, 220, 180, 240, 210],
                "HeartDisease": [0, 1, 0, 1, 1],
            }
        )

    @pytest.fixture
    def sample_df_without_sex(self):
        """
        Dataset missing Sex column.
        """

        return pd.DataFrame(
            {"Age": [45, 50, 55], "Cholesterol": [200, 220, 180], "HeartDisease": [0, 1, 0]}
        )

    def test_creates_all_three_splits(self, mock_paths, sample_df_with_sex):
        """
        Test that all three CSV files are created.
        """

        generate_gender_splits(sample_df_with_sex)

        assert (mock_paths / "female.csv").exists()
        assert (mock_paths / "male.csv").exists()
        assert (mock_paths / "nosex.csv").exists()

    def test_female_split_correct_rows(self, mock_paths, sample_df_with_sex):
        """
        Test that female split contains only Sex==0 rows.
        """

        generate_gender_splits(sample_df_with_sex)

        df_female = pd.read_csv(mock_paths / "female.csv")
        assert len(df_female) == 3
        assert all(df_female["Sex"] == 0)

    def test_male_split_correct_rows(self, mock_paths, sample_df_with_sex):
        """
        Test that male split contains only Sex==1 rows.
        """

        generate_gender_splits(sample_df_with_sex)

        df_male = pd.read_csv(mock_paths / "male.csv")
        assert len(df_male) == 2
        assert all(df_male["Sex"] == 1)

    def test_nosex_split_removes_sex_column(self, mock_paths, sample_df_with_sex):
        """
        Test that nosex split doesn't contain Sex column.
        """

        generate_gender_splits(sample_df_with_sex)

        df_nosex = pd.read_csv(mock_paths / "nosex.csv")
        assert "Sex" not in df_nosex.columns
        assert len(df_nosex) == 5

    def test_no_sex_column_in_input(self, mock_paths, sample_df_without_sex):
        """
        Test behavior when input DataFrame has no Sex column.
        """

        generate_gender_splits(sample_df_without_sex)

        assert (
            not (mock_paths / "female.csv").exists()
            or len(pd.read_csv(mock_paths / "female.csv")) == 0
        )

        df_nosex = pd.read_csv(mock_paths / "nosex.csv")
        assert len(df_nosex) == 3

    def test_empty_dataframe(self, mock_paths):
        """
        Test with an empty DataFrame.
        """

        df_empty = pd.DataFrame(columns=["Age", "Sex", "Cholesterol"])
        generate_gender_splits(df_empty)

        df_nosex = pd.read_csv(mock_paths / "nosex.csv")
        assert len(df_nosex) == 0

    def test_all_female_dataset(self, mock_paths):
        """
        Test with dataset containing only females.
        """

        df = pd.DataFrame({"Age": [45, 50, 55], "Sex": [0, 0, 0], "Cholesterol": [200, 220, 180]})

        generate_gender_splits(df)

        df_female = pd.read_csv(mock_paths / "female.csv")
        df_male = pd.read_csv(mock_paths / "male.csv")

        assert len(df_female) == 3
        assert len(df_male) == 0

    def test_preserves_all_columns_except_sex(self, mock_paths, sample_df_with_sex):
        """
        Test that all columns except Sex are preserved in nosex split.
        """

        generate_gender_splits(sample_df_with_sex)

        df_nosex = pd.read_csv(mock_paths / "nosex.csv")
        expected_cols = [col for col in sample_df_with_sex.columns if col != "Sex"]

        assert list(df_nosex.columns) == expected_cols


class TestPreprocessing:
    @pytest.fixture
    def raw_heart_data(self):
        """
        Fixture with heart disease dataset.
        """

        return pd.DataFrame(
            {
                "Age": [40, 50, 60, 45, 55],
                "Sex": ["M", "F", "M", "F", "M"],
                "ChestPainType": ["ATA", "NAP", "ASY", "ATA", "NAP"],
                "RestingBP": [120, 130, 0, 140, 125],
                "Cholesterol": [200, 0, 250, 0, 220],
                "MaxHR": [150, 160, 140, 155, 145],
                "ExerciseAngina": ["N", "Y", "N", "Y", "N"],
                "Oldpeak": [1.0, 2.0, 1.5, 2.5, 1.2],
                "ST_Slope": ["Up", "Flat", "Down", "Up", "Flat"],
                "RestingECG": ["Normal", "ST", "Normal", "LVH", "Normal"],
                "HeartDisease": [0, 1, 1, 0, 1],
            }
        )

    def test_preprocessing_removes_invalid_restingbp(self, mock_paths, raw_heart_data):
        """
        Test that rows with RestingBP==0 are removed.
        """

        (mock_paths / "raw").mkdir(parents=True, exist_ok=True)
        raw_heart_data.to_csv(mock_paths / "raw" / "heart.csv", index=False)

        df = preprocessing()

        assert len(df) == 4
        assert all(df["RestingBP"] != 0)

    @pytest.mark.parametrize(
        "sex_input,sex_expected",
        [
            ("M", 1),
            ("F", 0),
        ],
    )
    def test_preprocessing_encodes_sex(self, mock_paths, raw_heart_data, sex_input, sex_expected):
        """
        Test that Sex is correctly encoded as binary.
        """

        data = raw_heart_data.copy()
        data["Sex"] = sex_input
        data["RestingBP"] = 120  # BHOOO

        (mock_paths / "raw").mkdir(parents=True, exist_ok=True)
        data.to_csv(mock_paths / "raw" / "heart.csv", index=False)

        df = preprocessing()
        assert all(df["Sex"] == sex_expected)

    @pytest.mark.parametrize(
        "exercise_input,exercise_expected",
        [
            ("Y", 1),
            ("N", 0),
        ],
    )
    def test_preprocessing_encodes_exercise_angina(
        self, mock_paths, raw_heart_data, exercise_input, exercise_expected
    ):
        """
        Test that ExerciseAngina is correctly encoded as binary.
        """

        data = raw_heart_data.copy()
        data["ExerciseAngina"] = exercise_input
        data["RestingBP"] = 120  # BHOOO

        (mock_paths / "raw").mkdir(parents=True, exist_ok=True)
        data.to_csv(mock_paths / "raw" / "heart.csv", index=False)

        df = preprocessing()

        assert all(df["ExerciseAngina"] == exercise_expected)

    def test_preprocessing_one_hot_encoding(self, mock_paths, raw_heart_data):
        """
        Test that categorical features are encoded.
        """

        (mock_paths / "raw").mkdir(parents=True, exist_ok=True)
        raw_heart_data.to_csv(mock_paths / "raw" / "heart.csv", index=False)

        df = preprocessing()

        assert "ChestPainType" not in df.columns
        assert "RestingECG" not in df.columns
        assert "ST_Slope" not in df.columns

        assert any("ChestPainType_" in col for col in df.columns)
        assert any("RestingECG_" in col for col in df.columns)
        assert any("ST_Slope_" in col for col in df.columns)

    def test_preprocessing_scales_numerical_features(self, mock_paths, raw_heart_data):
        """
        Test that numerical features are scaled.
        """

        (mock_paths / "raw").mkdir(parents=True, exist_ok=True)
        raw_heart_data.to_csv(mock_paths / "raw" / "heart.csv", index=False)

        df = preprocessing()

        for col in ["Age", "MaxHR", "Oldpeak"]:
            if col in df.columns:
                assert abs(df[col].mean()) < 0.1
                assert abs(df[col].std() - 1.0) < 0.2

    def test_preprocessing_saves_scaler(self, mock_paths, raw_heart_data):
        """
        Test that the fitted scaler is saved.
        """

        (mock_paths / "raw").mkdir(parents=True, exist_ok=True)
        raw_heart_data.to_csv(mock_paths / "raw" / "heart.csv", index=False)

        preprocessing()

        assert (mock_paths / "artifacts" / "scaler.pkl").exists()

        scaler = joblib.load(mock_paths / "artifacts" / "scaler.pkl")
        assert isinstance(scaler, StandardScaler)
        assert hasattr(scaler, "mean_")

    def test_preprocessing_saves_csv(self, mock_paths, raw_heart_data):
        """
        Test that preprocessed CSV is saved.
        """

        (mock_paths / "raw").mkdir(parents=True, exist_ok=True)
        raw_heart_data.to_csv(mock_paths / "raw" / "heart.csv", index=False)

        preprocessing()

        assert (mock_paths / "preprocessed.csv").exists()

        df = pd.read_csv(mock_paths / "preprocessed.csv")
        assert len(df) > 0
        assert "HeartDisease" in df.columns

    def test_preprocessing_preserves_target(self, mock_paths, raw_heart_data):
        """
        Test that target column is preserved and not scaled.
        """

        (mock_paths / "raw").mkdir(parents=True, exist_ok=True)
        raw_heart_data.to_csv(mock_paths / "raw" / "heart.csv", index=False)

        df = preprocessing()

        assert "HeartDisease" in df.columns
        assert set(df["HeartDisease"].unique()).issubset({0, 1})

    def test_preprocessing_missing_raw_file(self, mock_paths):
        """
        Test that FileNotFoundError is raised when raw file is missing.
        """

        with pytest.raises(FileNotFoundError):
            preprocessing()

    def test_preprocessing_empty_dataset(self, mock_paths):
        """
        Test behavior with an empty dataset.
        """

        (mock_paths / "raw").mkdir(parents=True, exist_ok=True)

        empty_df = pd.DataFrame(
            columns=[
                "Age",
                "Sex",
                "ChestPainType",
                "RestingBP",
                "Cholesterol",
                "MaxHR",
                "ExerciseAngina",
                "Oldpeak",
                "ST_Slope",
                "RestingECG",
                "HeartDisease",
            ]
        )
        empty_df.to_csv(mock_paths / "raw" / "heart.csv", index=False)

        with pytest.raises(ValueError):  # , KeyError)):
            preprocessing()

    def test_preprocessing_returns_dataframe(self, mock_paths, raw_heart_data):
        """
        Test that preprocessing returns a pandas DataFrame.
        """

        (mock_paths / "raw").mkdir(parents=True, exist_ok=True)
        raw_heart_data.to_csv(mock_paths / "raw" / "heart.csv", index=False)

        result = preprocessing()

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_preprocessing_all_cholesterol_zero(self, mock_paths, raw_heart_data):
        """
        Test edge case where all Cholesterol values are 0, NaN returned.
        """

        data = raw_heart_data.copy()
        data["Cholesterol"] = 0

        (mock_paths / "raw").mkdir(parents=True, exist_ok=True)
        data.to_csv(mock_paths / "raw" / "heart.csv", index=False)

        df = preprocessing()

        assert df["Cholesterol"].isna().all()

    def test_preprocessing_target_as_integer(self, mock_paths, raw_heart_data):
        """
        Test that target is converted to integer type.
        """

        data = raw_heart_data.copy()
        data["HeartDisease"] = data["HeartDisease"].astype(float)

        (mock_paths / "raw").mkdir(parents=True, exist_ok=True)
        data.to_csv(mock_paths / "raw" / "heart.csv", index=False)

        df = preprocessing()

        assert df["HeartDisease"].dtype == np.int64 or df["HeartDisease"].dtype == np.int32

    @pytest.mark.xfail(
        reason="Known issue: preprocessing fails with single-row datasets due to scaling"
    )
    def test_preprocessing_single_row(self, mock_paths):
        """
        Test edge case preprocessing with only one row.
        """

        (mock_paths / "raw").mkdir(parents=True, exist_ok=True)

        single_row = pd.DataFrame(
            {
                "Age": [50],
                "Sex": ["M"],
                "ChestPainType": ["ATA"],
                "RestingBP": [120],
                "Cholesterol": [200],
                "MaxHR": [150],
                "ExerciseAngina": ["N"],
                "Oldpeak": [1.0],
                "ST_Slope": ["Up"],
                "RestingECG": ["Normal"],
                "HeartDisease": [0],
            }
        )
        single_row.to_csv(mock_paths / "raw" / "heart.csv", index=False)

        df = preprocessing()
        assert len(df) == 1


class TestPreprocessingIntegration:
    """
    Integration tests for the full preprocessing.
    """

    def test_full_pipeline_with_splits(self, mock_paths):
        data = pd.DataFrame(
            {
                "Age": [40, 50, 60, 45, 55, 35],
                "Sex": ["M", "F", "M", "F", "M", "F"],
                "ChestPainType": ["ATA", "NAP", "ASY", "ATA", "NAP", "ASY"],
                "RestingBP": [120, 130, 140, 135, 125, 128],
                "Cholesterol": [200, 210, 250, 230, 220, 205],
                "MaxHR": [150, 160, 140, 155, 145, 158],
                "ExerciseAngina": ["N", "Y", "N", "Y", "N", "Y"],
                "Oldpeak": [1.0, 2.0, 1.5, 2.5, 1.2, 1.8],
                "ST_Slope": ["Up", "Flat", "Down", "Up", "Flat", "Down"],
                "RestingECG": ["Normal", "ST", "Normal", "LVH", "Normal", "ST"],
                "HeartDisease": [0, 1, 1, 0, 1, 0],
            }
        )

        (mock_paths / "raw").mkdir(parents=True, exist_ok=True)
        data.to_csv(mock_paths / "raw" / "heart.csv", index=False)

        df = preprocessing()
        generate_gender_splits(df)

        assert (mock_paths / "preprocessed.csv").exists()
        assert (mock_paths / "female.csv").exists()
        assert (mock_paths / "male.csv").exists()
        assert (mock_paths / "nosex.csv").exists()
        assert (mock_paths / "artifacts" / "scaler.pkl").exists()

        df_female = pd.read_csv(mock_paths / "female.csv")
        df_male = pd.read_csv(mock_paths / "male.csv")

        assert len(df_female) == 3
        assert len(df_male) == 3
