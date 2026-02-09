import pandas as pd
import numpy as np


class MetricsCalculator:
    def calculate_all(self, df1, df2, columns):
        """Calculates Max diff, MAE, and RMSE for common columns."""
        results = []
        for col in columns:
            val1 = df1[col].values
            val2 = df2[col].values

            # Handle length mismatch by truncating to the shorter length
            min_len = min(len(val1), len(val2))
            if len(val1) != len(val2):
                print(
                    f"Warning: Length mismatch for {col}. Truncating to {min_len} rows."
                )

            val1 = val1[:min_len]
            val2 = val2[:min_len]

            diff = np.abs(val1 - val2)
            max_diff = np.max(diff)
            mae = np.mean(diff)
            rmse = np.sqrt(np.mean((val1 - val2) ** 2))

            results.append(
                {
                    "Column": col,
                    "Max Abs Difference": max_diff,
                    "MAE": mae,
                    "RMSE": rmse,
                }
            )

        return pd.DataFrame(results).set_index("Column")
