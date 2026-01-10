# Reports

This folder contains all the reports generated during the development and testing of the CardioTrack project, including test results, data validation reports, and monitoring figures.

## Folder Structure

```
reports/
├── figures/                        <- Monitoring charts and visualizations
├── great_expectations_reports/     <- Data quality validation reports
└── pytest_report/                  <- Unit and behavioral test report
```

## Unit and behavioral test report

### Behavioral Model Tests

This suite contains **12 tests** focused on validating the ML model behavior. It includes directional tests (verifying that predictions change as expected when features change), invariance tests (ensuring stability under certain transformations), and minimum functionality tests (checking basic model operations). All tests passed successfully.

### Project Tests

The main test suite covers **80 tests** across the entire codebase, including API endpoints, data preprocessing, model training, evaluation, and explainability features. All tests passed successfully.

## Data Validation Reports

Data quality is validated using Great Expectations. The HTML reports in `great_expectations_reports/` show validation results for both the raw and processed datasets, ensuring data integrity throughout the pipeline.

## Monitoring

### Uptime
The Better Stack dashboard displays real-time uptime monitoring for the CardioTrack API hosted on Hugging Face Spaces. The CardioTrack-Monitoring-Alert monitor performs health checks every 3 minutes and is configured to automatically notify the team via email when incidents occur.
![API Uptime Monitoring](figures/uptime.png)
