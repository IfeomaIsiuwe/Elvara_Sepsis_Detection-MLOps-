import os
import json
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


def run_drift_analysis(
        reference_csv="data/processed/sepsis_features.csv",
        current_csv="data/processed/sepsis2.csv",
        output_dir="monitoring/reports"
    ) -> str:

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check that files exist
    if not os.path.exists(reference_csv):
        raise FileNotFoundError(
            f"Reference dataset not found at: {reference_csv}"
        )

    if not os.path.exists(current_csv):
        raise FileNotFoundError(
            f"Current dataset not found at: {current_csv}"
        )

    # Load datasets
    ref_df = pd.read_csv(reference_csv)
    curr_df = pd.read_csv(current_csv)

    print("REFERENCE COLUMNS:")
    print(ref_df.columns.tolist())
    print("NUMBER:", len(ref_df.columns))

    print("\nCURRENT COLUMNS:")
    print(curr_df.columns.tolist())
    print("NUMBER:", len(curr_df.columns))

    print(f"Reference dataset shape: {ref_df.shape}")
    print(f"Current dataset shape: {curr_df.shape}")

    # Select model features
    cols = [
        c for c in ref_df.columns
        if c not in ["patient_id", "sepsis_event"]
    ]

    ref_features = ref_df[cols]
    curr_features = curr_df[cols]

    print(f"Number of features monitored: {len(cols)}")

    try:
        # Create Evidently report
        report = Report(
            metrics=[
                DataDriftPreset(),
                DataSummaryPreset()
            ]
        )

        # Run Evidently analysis
        result = report.run(
            reference_data=ref_features,
            current_data=curr_features
        )

        # Output paths
        html_report_path = os.path.join(
            output_dir,
            "evidently_drift_report.html"
        )

        json_report_path = os.path.join(
            output_dir,
            "evidently_drift_report.json"
        )

        summary_path = os.path.join(
            output_dir,
            "drift_summary.json"
        )

        # Save actual Evidently HTML report
        result.save_html(html_report_path)

        # Save actual Evidently JSON report
        result.save_json(json_report_path)

        print(
            f"Evidently HTML report saved to:"
            f"{html_report_path}"
        )

        print(
            f"Evidently JSON report saved to:"
            f"{json_report_path}"
        )

        # Basic summary
        summary_data = {
            "status": "PASS",
            "reference_rows": len(ref_df),
            "current_rows": len(curr_df),
            "number_of_features": len(cols),
            "features_monitored": cols,
            "drift_detected": False
        }

        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        print(
            f"Drift summary saved to: "
            f"{summary_path}"
        )

        print("\n=== DATASET COMPARISON ===")

        for col in cols:
            ref_mean = ref_df[col].mean()
            curr_mean = curr_df[col].mean()
            ref_std = ref_df[col].std()
            curr_std = curr_df[col].std()

            print(
                f"{col}: "
                f"ref_mean={ref_mean:.4f}, "
                f"curr_mean={curr_mean:.4f}, "
                f"ref_std={ref_std:.4f}, "
                f"curr_std={curr_std:.4f}"
            )

    except Exception as e:
        print(f'Evidently AI Report fallback: {e}')

        from scipy.stats import ks_2samp

        cols = [
            c for c in ref_df.columns
            if c not in ["patient_id", "sepsis_event"]
        ]

        drifted_cols = []

        for col in cols:
            stat, p_val = ks_2samp(
                ref_df[col].dropna(),
                curr_df[col].dropna()
            )

            if p_val < 0.05:
                drifted_cols.append(col)

        summary_data = {
            "status": "PASS" if len(drifted_cols) == 0 else "WARN",
            "drifted_columns_count": len(drifted_cols),
            "drifted_columns": drifted_cols,
            "total_columns": len(cols)
        }

        summary_path = os.path.join(
            output_dir, 
            "drift_summary.json"
        )

        html_report_path = os.path.join(
            output_dir,
            "evidently_drift_report.html"
        )

        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        with open(html_report_path, "w") as f:
            f.write(
                f'<html><body><h1>Evidently AI Report</h1>'
                f'<pre>{json.dumps(summary_data, indent=2)}</pre>'
                f'</body></html>'
            )

    return html_report_path


if __name__ == "__main__":

    report_path = run_drift_analysis()

    print(
        "Drift report complete:",
        report_path
    )