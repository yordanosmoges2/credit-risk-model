import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data from a given file path with basic error handling.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")


def basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return basic descriptive statistics for a dataframe.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    return df.describe(include="all")
