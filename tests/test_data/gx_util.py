import great_expectations as gx


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
