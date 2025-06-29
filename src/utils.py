"""Utility functions for credit risk modeling."""

def log(message: str):
    print(message)
def load_data(filePath: str):
    """Load data from a specified file path."""
    import pandas as pd
    try:
        data = pd.read_csv(filePath)
        log(f"Data loaded successfully from {filePath}")
        return data
    except Exception as e:
        log(f"Error loading data from {filePath}: {e}")
        return None
def saveData(data, filePath: str):
    """Save data to a specified file path."""
    try:
        data.to_csv(filePath, index=False)
        log(f"Data saved successfully to {filePath}")
    except Exception as e:
        log(f"Error saving data to {filePath}: {e}")