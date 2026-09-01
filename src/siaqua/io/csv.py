import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_FOLDER = PROJECT_ROOT / "data"



def read_csv(filename : str, data_type : str) -> pd.DataFrame:
    if data_type not in ("raw", "processed", "interim"):
        raise ValueError("Invalid data type. use 'raw', 'processed', or 'interim'")
    df = pd.read_csv(DATA_FOLDER / data_type / filename)
    return df


def save_dataframe(df : pd.DataFrame, data_type : str, name : str):
    if data_type not in ("raw", "processed", "interim"):
            raise ValueError("Invalid data type. use 'raw', 'processed', or 'interim'")
    file_path = DATA_FOLDER / data_type / name
    df.to_csv(file_path)
    logger.info("Dataframe saved succesfully: %s", file_path)

    