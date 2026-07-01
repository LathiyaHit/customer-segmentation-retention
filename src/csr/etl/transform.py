"""
Transform stage — clean the raw DataFrame into an analysis-ready shape.

Cleaning Steps:
parse dates -> drop null CustomerID -> strip cancellations ->
drop non-positive Quantity/UnitPrice -> drop duplicates ->
cap outliers at the 99th percentile -> compute Revenue -> cast dtypes.
"""

import sys

import pandas as pd

from csr.constants import constants
from csr.exception.exception import CSRException
from csr.logging.logger import logging


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw Online Retail DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data straight out of extract.extract_raw_data().

    Returns
    -------
    pd.DataFrame
        Cleaned data — no nulls, no cancellations, no non-positive
        Quantity/UnitPrice, no duplicates, outliers capped, plus a
        derived `Revenue` column.
    """
    try:
        logging.info(f"Transform stage started | input shape={df.shape}")
        data = df.copy()

        # 1. Parse dates
        data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])

        # 2. Drop missing CustomerID, normalise to string
        before = len(data)
        data = data.dropna(subset=["CustomerID"])
        data["CustomerID"] = data["CustomerID"].astype(int).astype(str)
        logging.info(f"Dropped {before - len(data)} rows with null CustomerID")

        # 3. Remove cancellations (InvoiceNo starting with 'C')
        before = len(data)
        data = data[
            ~data["InvoiceNo"].astype(str).str.startswith(constants.CANCELLATION_PREFIX)
        ]
        logging.info(f"Dropped {before - len(data)} cancellation rows")

        # 4. Remove non-positive Quantity / UnitPrice
        before = len(data)
        data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]
        logging.info(
            f"Dropped {before - len(data)} rows with zero/negative Quantity or UnitPrice"
        )

        # 5. Remove duplicate rows
        before = len(data)
        data = data.drop_duplicates()
        logging.info(f"Dropped {before - len(data)} duplicate rows")

        # 6. Cap outliers at the 99th percentile
        q99_qty = data["Quantity"].quantile(constants.OUTLIER_QUANTILE)
        q99_price = data["UnitPrice"].quantile(constants.OUTLIER_QUANTILE)
        before = len(data)
        data = data[(data["Quantity"] <= q99_qty) & (data["UnitPrice"] <= q99_price)]
        logging.info(
            f"Outlier cap: Qty<={q99_qty:.0f}, Price<={q99_price:.2f} | "
            f"dropped {before - len(data)} rows"
        )

        # 7. Derived Revenue column
        data["Revenue"] = data["Quantity"] * data["UnitPrice"]

        # 8. Standardise dtypes on key string columns
        for col in constants.STRING_COLUMNS:
            data[col] = data[col].astype(str)

        # 9. Final validation — matches the notebook's assertion cell
        if data.isnull().sum().sum() != 0:
            raise ValueError("Nulls remain after cleaning")
        if not (data["Quantity"] > 0).all():
            raise ValueError("Non-positive quantities remain after cleaning")
        if not (data["UnitPrice"] > 0).all():
            raise ValueError("Non-positive prices remain after cleaning")

        logging.info(
            f"Transform stage completed | output shape={data.shape} | "
            f"customers={data['CustomerID'].nunique()} | "
            f"orders={data['InvoiceNo'].nunique()} | "
            f"revenue=£{data['Revenue'].sum():,.2f}"
        )
        return data

    except Exception as e:
        logging.error(f"Transform stage failed: {e}")
        raise CSRException(e, sys) from e


if __name__ == "__main__":
    from csr.etl.extract import extract_raw_data

    raw = extract_raw_data()
    cleaned = transform_data(raw)
    print(cleaned.head())
    print(f"Shape: {cleaned.shape}")