import json

import great_expectations as gx
from loguru import logger


def set_gx(source_name, asset_name, suite_name):
    context = gx.get_context()

    data_source = context.data_sources.add_pandas(name=source_name)
    data_asset = data_source.add_dataframe_asset(name=asset_name)

    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch_definition")

    suite = context.suites.add(
        gx.core.expectation_suite.ExpectationSuite(
            name=suite_name,
        )
    )
    return context, suite, batch_definition


def show_results(checkpoint_result):
    result = json.loads(checkpoint_result.describe())

    for i, v in enumerate(result.get("validation_results", []), 1):
        logger.info(f"Validation {i}, {v['success']}")

        for e in v.get("expectations", []):
            name = e["expectation_type"]
            kwargs = e.get("kwargs", {})
            col = kwargs.get("column")
            success = e["success"]
            logger.info(f"{col}: {name} -> {success}")

            if not success:
                logger.warning(kwargs)
                logger.warning(e.get("result"))

    if result.get("success"):
        logger.success(f"Overall success: {result.get('success')}")
    else:
        logger.error(f"Overall success: {result.get('success')}")
