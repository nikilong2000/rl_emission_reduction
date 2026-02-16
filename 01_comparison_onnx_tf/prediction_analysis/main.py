import argparse
import os
import sys

# Add the current directory to sys.path to allow imports when running from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from metrics import MetricsCalculator
from plotting import ComparatorPlotter


def main():

    df1_filename = "old_model_predictions(2).csv"
    df2_filename = "onnx_model_predictions(2).csv"

    parser = argparse.ArgumentParser(
        description="Compare two CSV files of model predictions."
    )
    parser.add_argument(
        "--file1",
        type=str,
        default=os.path.join("performance_measurements", df1_filename),
        help="Path to reference CSV file",
    )
    parser.add_argument(
        "--file2",
        type=str,
        default=os.path.join("performance_measurements", df2_filename),
        help="Path to comparison CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="prediction_analysis_results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Ensure output dir exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    loader = DataLoader()
    metrics_calc = MetricsCalculator()
    plotter = ComparatorPlotter()

    print(f"Loading files:\n  1: {args.file1}\n  2: {args.file2}")
    try:
        df1 = loader.load_csv(args.file1)
        df2 = loader.load_csv(args.file2)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    common_columns = loader.validate_columns(df1, df2)

    if not common_columns:
        print("No common columns found to compare.")
        return

    print(f"Comparing {len(common_columns)} columns...")

    # Calculate metrics
    metrics_df = metrics_calc.calculate_all(df1, df2, common_columns)
    print("\nMetrics Summary:")
    print(metrics_df)

    metrics_path = os.path.join(args.output_dir, "comparison_metrics.csv")
    metrics_df.to_csv(metrics_path)
    print(f"\nMetrics saved to {metrics_path}")

    # Generate plots
    plotter.plot_comparisons(
        df1, df2, df1_filename, df2_filename, common_columns, args.output_dir
    )


if __name__ == "__main__":
    main()
