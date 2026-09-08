"""Analyze saved batch results without rerunning the simulation."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "results" / "raw" / "smoke_test.csv"
SUMMARY_DIRECTORY = PROJECT_ROOT / "results" / "summary"
SUMMARY_PATH = SUMMARY_DIRECTORY / "smoke_test_summary.csv"
AGGREGATE_PATH = (SUMMARY_DIRECTORY / "smoke_test_aggregate.csv")
TIME_SERIES_PATH = (SUMMARY_DIRECTORY / "smoke_test_time_series.csv")

def main():
    """Load and inspect the saved smoke-test results."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Batch result file not found: {INPUT_PATH}")

    results_df = pd.read_csv(INPUT_PATH)

    columns_to_show = [
        "RunId",
        "Step",
        "seed",
        "simulated_time_s",
        "residual_deficit",
        "normalized_deficit",
        "satisfied_points",
    ]

    print(f"Loaded rows: {len(results_df)}")
    print(results_df[columns_to_show].head())
    print("...")
    print(results_df[columns_to_show].tail())

    # J_delta is defined over simulation steps 1, ..., T.
    # The initial snapshot at Step 0 is therefore excluded.
    evaluation_rows = results_df[results_df["Step"] > 0]

    # Sort chronologically before using "last" for final metrics.
    evaluation_rows = evaluation_rows.sort_values(["RunId", "Step"])

    # Produce one summary row for each independent model execution.
    run_summary_df = (
        evaluation_rows
        .groupby("RunId", as_index=False)
        .agg(
            seed=("seed", "first"),
            J_delta=("normalized_deficit", "mean"),
            final_time_s=("simulated_time_s", "last"),
            final_residual_deficit=("residual_deficit", "last"),
            final_normalized_deficit=("normalized_deficit", "last"),
            final_satisfied_points=("satisfied_points", "last"),
            final_active_points=("active_points", "last"),
            final_overservice=("overservice", "last"),
        )
    )

    # Normalize the number of satisfied points because dynamic runs may have different number of active points.
    active_points = run_summary_df["final_active_points"]

    run_summary_df["final_satisfied_fraction"] = (
        run_summary_df["final_satisfied_points"]
        .div(active_points)
        .where(active_points > 0)
    )

    
    # Calculate r_delta(t) separately within each replication.
    results_df = results_df.sort_values(["RunId", "Step"]).copy()

    previous_normalized_deficit = (
        results_df
        .groupby("RunId")["normalized_deficit"]
        .shift(1)
    )

    results_df["r_delta"] = (previous_normalized_deficit - results_df["normalized_deficit"])

    # Normalize satisfied points at every step because the number of active
    # points may change during dynamic scenarios.
    active_points_by_step = results_df["active_points"]

    results_df["satisfied_fraction"] = (
        results_df["satisfied_points"]
        .div(active_points_by_step)
        .where(active_points_by_step > 0)
    )



    # Aggregate the same simulation step across independent replications.
    # Step 0 is retained because it represents the initial state.
    time_series_summary_df = (
        results_df
        .groupby("Step", as_index=False)
        .agg(
            simulated_time_s=("simulated_time_s", "first"),
            replications=("RunId", "nunique"),
            normalized_deficit_mean=("normalized_deficit", "mean"),
            normalized_deficit_std=("normalized_deficit", "std"),
            r_delta_mean=("r_delta", "mean"),
            r_delta_std=("r_delta", "std"),
            satisfied_fraction_mean=("satisfied_fraction", "mean"),
            satisfied_fraction_std=("satisfied_fraction", "std"),
            residual_deficit_mean=("residual_deficit", "mean"),
            residual_deficit_std=("residual_deficit", "std"),
            overservice_mean=("overservice", "mean"),
            overservice_std=("overservice", "std"),
            idle_drones_mean=("idle_drones", "mean"),
            idle_drones_std=("idle_drones", "std"),
            exploring_drones_mean=("exploring_drones", "mean"),
            exploring_drones_std=("exploring_drones", "std"),
            active_points_mean=("active_points", "mean"),
            active_points_std=("active_points", "std"),
            total_demand_mean=("total_demand", "mean"),
            total_demand_std=("total_demand", "std"),
            unavoidable_deficit_mean=("unavoidable_deficit", "mean"),
            unavoidable_deficit_std=("unavoidable_deficit", "std")
        )
    )

    # Aggregate the independent replications of the same configuration.
    # ddof=1 computes the sample standard deviation.
    aggregate_summary_df = pd.DataFrame(
        [
            {
                "replications": len(run_summary_df),
                "J_delta_mean": run_summary_df["J_delta"].mean(),
                "J_delta_std": run_summary_df["J_delta"].std(ddof=1),
                "final_residual_deficit_mean": (
                    run_summary_df["final_residual_deficit"].mean()
                ),
                "final_residual_deficit_std": (
                    run_summary_df["final_residual_deficit"].std(ddof=1)
                ),
                "final_normalized_deficit_mean": (
                    run_summary_df["final_normalized_deficit"].mean()
                ),
                "final_normalized_deficit_std": (
                    run_summary_df["final_normalized_deficit"].std(ddof=1)
                ),
                "final_satisfied_fraction_mean": (
                    run_summary_df["final_satisfied_fraction"].mean()
                ),
                "final_satisfied_fraction_std": (
                    run_summary_df["final_satisfied_fraction"].std(ddof=1)
                ),
                "final_overservice_mean": (
                    run_summary_df["final_overservice"].mean()
                ),
                "final_overservice_std": (
                    run_summary_df["final_overservice"].std(ddof=1)
                ),
            }
        ]
    )

    # Save one compact summary row for each model execution.
    SUMMARY_DIRECTORY.mkdir(parents=True, exist_ok=True)
    run_summary_df.to_csv(SUMMARY_PATH, index=False)
    
    aggregate_summary_df.to_csv(AGGREGATE_PATH, index=False)

    time_series_summary_df.to_csv(TIME_SERIES_PATH, index=False)
    
    print()
    print("Summary by run:")
    print(run_summary_df)
    print(f"Summary saved to: {SUMMARY_PATH}")
    
    print()
    print("Aggregate summary:")
    print(aggregate_summary_df)
    print(f"Aggregate summary saved to: {AGGREGATE_PATH}")

    print()
    print("Time-series summary:")
    print(time_series_summary_df)
    print(f"Time-series summary saved to: {TIME_SERIES_PATH}")
    
if __name__ == "__main__":
    main()