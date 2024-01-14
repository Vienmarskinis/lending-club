import pandas as pd


def extract_year(df: pd.DataFrame) -> pd.DataFrame:
    """Extract year from a dataframe columns in %b-%Y format"""
    columns = df.columns
    for col in columns:
        df[f"{col}_month"] = pd.to_datetime(df[col], format="%b-%Y").dt.year
    return df.drop(columns=columns)


def year_pipe_names_out(self, cols):
    """quick workaround for feature_names_out"""
    return cols
