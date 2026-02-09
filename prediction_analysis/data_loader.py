import pandas as pd
import os

class DataLoader:
    def load_csv(self, path):
        """Loads a CSV file into a DataFrame."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path} (cwd: {os.getcwd()})")
        return pd.read_csv(path)

    def validate_columns(self, df1, df2):
        """Checks intersection of columns and returns valid common columns list."""
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        
        common_cols = list(cols1.intersection(cols2))
        missing_in_2 = cols1 - cols2
        missing_in_1 = cols2 - cols1
        
        if missing_in_2:
            print(f"Warning: Columns in file 1 but not in file 2: {missing_in_2}")
        if missing_in_1:
            print(f"Warning: Columns in file 2 but not in file 1: {missing_in_1}")
            
        return sorted(common_cols)
