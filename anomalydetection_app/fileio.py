import base64
import io

import pandas as pd
from pyarrow import parquet as pa_parquet


def columns_from_octet_stream(content_string, file_name):
    decoded = base64.b64decode(content_string)
    options = []
    if '.parquet' in file_name:
        options = columns_from_parquet_file(io.BytesIO(decoded))
    if '.csv' in file_name:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), nrows=5)
        options = [{'label': col, 'value': col} for col in df.columns]
    return options


def columns_from_parquet_file(input_parquet):
    schema = pa_parquet.read_schema(input_parquet)
    column_specifications = schema.pandas_metadata['columns']
    options = []
    for col_item in column_specifications:
        name = col_item['name']
        if name is not None:
            options.append({'label': name, 'value': name})
    return options
