"""
Extract stage — load the raw Online Retail CSV into memory.
"""
import sys
from pathlib import Path

import pandas as pd

from csr.config.configuration import get_config
from csr.constants import constants
from csr.exception.exception import CSRException
from csr.logging.logger import logging

def extract_raw_data(raw_csv_path: Path | None = None) -> pd.DataFrame:
    """
    Read the raw Online Retail CSV from disk.

    Parameters
    ----------
    raw_csv_path : Path, optional
        Override for the raw CSV location. Defaults to
        config.paths.raw_csv (data/raw/Online_Retail.csv).

    Returns
    -------
    pd.DataFrame
        Untouched raw data, exactly as it exists on disk.
    """
    try:
        config = get_config()
        path = raw_csv_path or config.paths.raw_csv

        logging.info(f"Extract stage started | reading: {path}")

        if not path.exists():
            raise FileNotFoundError(f"Raw data file not found at: {path}")

        df = pd.read_csv(path, encoding=constants.RAW_ENCODING)

        missing_cols = set(constants.RAW_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Raw CSV is missing expected columns: {sorted(missing_cols)}"
            )

        logging.info(
            f"Extract stage completed | shape={df.shape} | "
            f"columns={df.columns.tolist()}"
        )
        return df

    except Exception as e:
        logging.error(f"Extract stage failed: {e}")
        raise CSRException(e, sys) from e


if __name__ == "__main__":
    data = extract_raw_data()
    print(data.head())
    print(f"Shape: {data.shape}")