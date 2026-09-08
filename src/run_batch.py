"""Run reproducible batch experiments for CoverageModel."""

from pathlib import Path

import pandas as pd
from mesa.batchrunner import batch_run

from Model import CoverageModel



def main():
    """Run a minimal static batch experiment."""
    simulation_steps = 20

    parameters = {
        "width": 100.0,
        "height": 100.0,
        "n_drones": 40,
        "n_points": 12,
        "max_priority": 3,
        "point_margin": 0.0,
        "flight_buffer": None,
        "point_layout": "random",
        "point_routine": "static",
        "event_seed": 0,
        "deployment": "dispersed",
        "deployment_noise": 1.0,
        "drone_type": "quadcopter",
        "meters_per_unit": 1.0,
        "seconds_per_step": 1.0,
        "speed": 1.0,
        "drone_sensing_radius": 10.0,
        "point_sensing_radius": 10.0,
        "separation": 2.0,
        "coverage_radius": 8.0,
        "cohere": 0.25,
        "separate": 0.015,
        "match": 0.05,
        "boundary": 1.2,
        "margin": 12.0,
        "quadcopter_margin": 2.0,
        "beta": 0.05,
        "explore": 0.2,
        "release_delay_max_steps": 5,
        "avoid_angle_degrees": 10.0,
        "support_inset": 2.0,
        "collect_agent_data": False,
    }

    # Mesa creates one model execution for each seed.
    seeds = list(range(5))

    results = batch_run(
        CoverageModel,
        parameters=parameters,
        rng=seeds,
        max_steps=simulation_steps,
        data_collection_period=1,
        number_processes=1,
        display_progress=True,
    )

    # batch_run returns a list of dictionaries.
    # A DataFrame organizes them as rows and columns for analysis and saving.
    results_df = pd.DataFrame(results)

    # Verify that the columns required by the analysis are present.
    required_columns = {
        "RunId",
        "Step",
        "seed",
        "meters_per_unit",
        "seconds_per_step",
        "speed",
        "simulated_time_s",
        "residual_deficit",
        "normalized_deficit",
    }

    missing_columns = required_columns - set(results_df.columns)

    if missing_columns:
        raise RuntimeError(f"Batch results are missing columns: {sorted(missing_columns)}")

    expected_runs = len(seeds)
    actual_runs = results_df["RunId"].nunique()

    if actual_runs != expected_runs:
        raise RuntimeError(f"Expected {expected_runs} runs, but found {actual_runs}.")

    expected_rows_per_run = simulation_steps + 1
    rows_per_run = results_df.groupby("RunId").size()

    if not rows_per_run.eq(expected_rows_per_run).all():
        raise RuntimeError("One or more runs contain an unexpected number of rows.")

    # The physical time must correspond to the step and its duration.
    expected_time_s = (results_df["Step"] * results_df["seconds_per_step"])

    time_is_consistent = (results_df["simulated_time_s"] - expected_time_s).abs().le(1e-9).all()

    if not time_is_consistent:
        raise RuntimeError("Step and simulated physical time are inconsistent.")

    print("Batch result validation passed.")

    # Store generated data outside src/.
    project_root = Path(__file__).resolve().parents[1]
    raw_results_directory = project_root / "results" / "raw"

    # Create results/raw/ if it does not exist yet.
    raw_results_directory.mkdir(parents=True, exist_ok=True)

    output_path = raw_results_directory / "smoke_test.csv"

    # index=False prevents Pandas from adding an unnecessary row-number column.
    results_df.to_csv(output_path, index=False)

    print(f"Results saved to: {output_path}")

    columns_to_show = [
        "RunId",
        "Step",
        "seed",
        "simulated_time_s",
        "residual_deficit",
        "normalized_deficit",
        "satisfied_points",
    ]

    print(f"Collected rows: {len(results_df)}")
    print(results_df[columns_to_show])

if __name__ == "__main__":
    main()