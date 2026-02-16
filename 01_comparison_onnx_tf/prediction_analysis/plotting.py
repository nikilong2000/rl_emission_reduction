import matplotlib.pyplot as plt
import os


class ComparatorPlotter:
    def plot_comparisons(
        self, df1, df2, df1_filename, df2_filename, columns, output_dir
    ):
        """Generates line plots for each column comparing the two dataframes."""
        plots_dir = os.path.join(output_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        print(f"Generating plots in {plots_dir}...")

        for col in columns:
            plt.figure(figsize=(10, 6))

            # Align lengths
            min_len = min(len(df1), len(df2))

            plt.plot(df1[col].iloc[:min_len], label=df1_filename, alpha=0.7)
            plt.plot(
                df2[col].iloc[:min_len],
                label=df2_filename,
                alpha=0.7,
                linestyle="--",
            )

            plt.title(f"Comparison: {col}")
            plt.xlabel("Index (Time/Step)")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)

            # Sanitize filename
            safe_col = "".join([c if c.isalnum() else "_" for c in col])
            output_path = os.path.join(plots_dir, f"{safe_col}.png")
            plt.savefig(output_path)
            plt.close()

        print("Plots generated.")
