
import pandas as pd
import os

def load_data_file(file_path: str) -> pd.DataFrame:
    """Load the content of the uploaded CSV or Excel file."""
    file_extension = os.path.splitext(file_path)[1]
    
    if file_extension == ".csv":
        data = pd.read_csv(file_path)
    elif file_extension in [".xls", ".xlsx"]:
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Perform basic data cleaning such as handling missing values, and dropping duplicates."""
    # Example basic cleaning: Remove rows with missing values and duplicates.
    data_cleaned = data.dropna().drop_duplicates()
    return data_cleaned
