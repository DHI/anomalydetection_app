import pandas as pd
import os

from anomalydetection_app.fileio import columns_from_parquet_file


def test_columns_from_parquet_file(tmpdir):
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    parquet_file = os.path.join(tmpdir, 'df.parquet')
    df.to_parquet(parquet_file)
    options = columns_from_parquet_file(parquet_file)
    expected_options = [{'value': 'col1', 'label': 'col1'}, {'value': 'col2', 'label': 'col2'}]
    assert options == expected_options
