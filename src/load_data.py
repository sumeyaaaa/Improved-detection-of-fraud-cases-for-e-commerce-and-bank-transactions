import pandas as pd

def load_data(path):
    """
    Load a CSV file into a DataFrame.

    Parameters:
    path (str): The full path to the CSV file.

    Returns:
    pd.DataFrame: Loaded data as a DataFrame.
    """
    try:
        df = pd.read_csv(path)
        print(f"Data loaded successfully from {path}")
        return df
    except FileNotFoundError:
        print(f"File not found: {path}")
    except Exception as e:
        print(f"Error loading data: {e}")
