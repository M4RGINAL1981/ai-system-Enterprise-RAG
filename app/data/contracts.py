import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema

ReglamentChunkSchema = DataFrameSchema (
    {
        "chunk_id": Column(str, Check.str_matches(r"^chunk_\d{4}$"), nullable=False),
        "document_name": Column(str, Check.str_length(min_value=1), nullable=False),
        "page_number": Column(int, Check.in_range(1, 172), nullable=False),
        "access_group": Column(str, Check.isin(["client", "staff"]), nullable=False),
        "text": Column(str, Check.str_length(min_value=20), nullable=False),
    },
    strict=True,
    coerce=True,
)