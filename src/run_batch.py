"""Run reproducible batch experiments for CoverageModel."""
import json
from math import prod
from pathlib import Path

import pandas as pd
from mesa.batchrunner import batch_run

from Model import CoverageModel
from PointScenario import get_point_routine

def main():
    """Run a static coverage-radius pilot experiment."""
    experiment_name = "static_deployment_comparison"
    simulation_steps = 600

    data_collection_period = 1
    number_processes = None

    parameters = {
        "width": 100.0,
        "height": 100.0,
        "n_drones": 40,
        "n_points": 12,
        "min_priority": 1,
        "max_priority": 3,
        "point_margin": 0.0,
        "flight_buffer": None,
        "point_layout": "random",
        "point_routine": "static",
        "event_seed": 0,
        "deployment": ["dispersed", "base", "left"],
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

    # Keep one high-level marker for each environmental event.
    point_routine_definition = get_point_routine(parameters["point_routine"])

    event_markers = [
        {
            "step": int(event["step"]),
            "simulated_time_s": float(
                event["step"] * parameters["seconds_per_step"]
            ),
            "event_type": event["type"],
        }
        for event in point_routine_definition
    ]

    # Mesa creates one model execution for each configuration and seed.
    seeds = list(range(100))

    # Record everything needed to identify and reproduce the experiment.
    experiment_config = {
        "experiment_name": experiment_name,
        "simulation_steps": simulation_steps,
        "data_collection_period": data_collection_period,
        "number_processes": number_processes,
        "seeds": seeds,
        "parameters": parameters,
        "point_routine_definition": point_routine_definition,
        "event_markers": event_markers,
    }

    # Refuse to overwrite an experiment that has already been saved.
    project_root = Path(__file__).resolve().parents[1]
    experiment_directory = project_root / "results" / experiment_name
    output_path = experiment_directory / "raw_results.csv"
    config_path = experiment_directory / "experiment_config.json"

    if experiment_directory.exists():
        raise FileExistsError(
            f"Experiment directory already exists: {experiment_directory}. "
            "Choose a new experiment name or remove the existing directory explicitly."
        )

    results = batch_run(
        CoverageModel,
        parameters=parameters,
        rng=seeds,
        max_steps=simulation_steps,
        data_collection_period=data_collection_period,
        number_processes=number_processes,
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
        "n_drones",
        "meters_per_unit",
        "seconds_per_step",
        "speed",
        "simulated_time_s",
        "residual_deficit",
        "normalized_deficit",
        "underserved_points",
        "exactly_satisfied_points",
        "overserved_points",
        "idle_drones",
        "exploring_drones",
        "stationing_drones",
        "active_points",
        "overlapping_zones",
        "total_demand",
        "unavoidable_deficit",
    }

    missing_columns = required_columns - set(results_df.columns)

    if missing_columns:
        raise RuntimeError(f"Batch results are missing columns: {sorted(missing_columns)}")

    varying_parameter_sizes = [len(value) for value in parameters.values() if isinstance(value, list)]
    expected_configurations = prod(varying_parameter_sizes)
    expected_runs = len(seeds) * expected_configurations
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

    # Create the experiment directory only after the batch has completed
    # and its results have passed validation.
    experiment_directory.mkdir(parents=True)

    # index=False prevents Pandas from adding an unnecessary row-number column.
    results_df.to_csv(output_path, index=False)

    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(experiment_config, config_file, indent=4)

    print(f"Results saved to: {output_path}")
    print(f"Experiment configuration saved to: {config_path}")

    columns_to_show = [
        "RunId",
        "Step",
        "seed",
        "simulated_time_s",
        "residual_deficit",
        "normalized_deficit",
        "underserved_points",
        "exactly_satisfied_points",
        "overserved_points",
    ]

    print(f"Collected rows: {len(results_df)}")
    print(results_df[columns_to_show])

if __name__ == "__main__":
    main()
