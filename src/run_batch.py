"""Run reproducible batch experiments for CoverageModel."""
import json
from pathlib import Path

import pandas as pd
from mesa.batchrunner import batch_run

from Model import CoverageModel
from PointScenario import get_point_routine


def main():
    """Run E26b: reduce the interval between composite events to 150 s."""
    experiment_name = (
        "E26b_dinamico_frequenza_eventi_150s"
    )
    simulation_steps = 1200

    data_collection_period = 1
    number_processes = None

    parameters = {
        "width": 100.0,
        "height": 100.0,
        "n_drones": 20,
        "n_points": 10,
        "min_priority": 2,
        "max_priority": 2,
        "point_layout": "random",
        "point_margin": 0.0,
        "flight_buffer": None,
        "point_routine": "dynamic_composite_stress_150",
        "event_seed": 0,
        "deployment": "left",
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

    # These two linked profiles avoid generating the invalid cross-product (10, 25) and the communication-only profile (25, 10). 
    # E26 compares the nominal and enhanced co-visible configurations already used in E22/E23b.
    radius_profiles = {
        "rd10_rp10_nominal": {
            "drone_sensing_radius": 10.0,
            "point_sensing_radius": 10.0,
        },
        "rd25_rp25_potenziato": {
            "drone_sensing_radius": 25.0,
            "point_sensing_radius": 25.0,
        },
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
        "radius_profiles": radius_profiles,
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

    result_frames = []
    next_run_id = 1

    for radius_profile, radius_parameters in radius_profiles.items():
        configuration_parameters = {
            **parameters,
            **radius_parameters,
        }
        configuration_results = batch_run(
            CoverageModel,
            parameters=configuration_parameters,
            rng=seeds,
            max_steps=simulation_steps,
            data_collection_period=data_collection_period,
            number_processes=number_processes,
            display_progress=True,
        )
        configuration_df = pd.DataFrame(configuration_results)

        # Each batch_run call restarts Mesa's RunId sequence.
        # Remap it so the combined CSV still identifies every independent execution uniquely.
        original_run_ids = sorted(configuration_df["RunId"].unique())
        run_id_map = {
            original_run_id: next_run_id + offset
            for offset, original_run_id in enumerate(original_run_ids)
        }
        configuration_df["RunId"] = configuration_df["RunId"].map(run_id_map)
        configuration_df["radius_profile"] = radius_profile
        next_run_id += len(original_run_ids)
        result_frames.append(configuration_df)

    results_df = pd.concat(result_frames, ignore_index=True)

    # Verify that the columns required by the analysis are present.
    required_columns = {
        "RunId",
        "Step",
        "seed",
        "radius_profile",
        "n_drones",
        "n_points",
        "min_priority",
        "max_priority",
        "drone_type",
        "drone_sensing_radius",
        "point_sensing_radius",
        "coverage_radius",
        "flight_buffer",
        "cohere",
        "match",
        "meters_per_unit",
        "seconds_per_step",
        "speed",
        "simulated_time_s",
        "residual_deficit",
        "normalized_deficit",
        "underserved_points",
        "exactly_satisfied_points",
        "overserved_points",
        "excess_service_units",
        "idle_drones",
        "exploring_drones",
        "stationing_drones",
        "active_points",
        "overlapping_zones",
        "total_demand",
        "unavoidable_deficit",
        "point_layout",
        "point_routine",
    }

    missing_columns = required_columns - set(results_df.columns)

    if missing_columns:
        raise RuntimeError(f"Batch results are missing columns: {sorted(missing_columns)}")

    expected_configurations = len(radius_profiles)
    expected_runs = expected_configurations * len(seeds)
    actual_runs = results_df["RunId"].nunique()

    if actual_runs != expected_runs:
        raise RuntimeError(f"Expected {expected_runs} runs, but found {actual_runs}.")

    if not (
        results_df["n_drones"].eq(20)
        & results_df["n_points"].eq(10)
        & results_df["min_priority"].eq(2)
        & results_df["max_priority"].eq(2)
        & results_df["point_layout"].eq("random")
        & results_df["point_routine"].eq("dynamic_composite_stress_150")
        & results_df["drone_type"].eq("quadcopter")
        & results_df["drone_sensing_radius"].isin([10.0, 25.0])
        & results_df["point_sensing_radius"].isin([10.0, 25.0])
        & results_df["radius_profile"].isin(radius_profiles)
        & results_df["coverage_radius"].eq(8.0)
        & results_df["flight_buffer"].isna()
        & results_df["cohere"].eq(0.25)
        & results_df["match"].eq(0.05)
    ).all():
        raise RuntimeError(
            "E26b requires 20 drones, 10 points with uniform quota 2, "
            "point_layout = 'random', the 150-second composite routine, "
            "quadcopters, radius profiles (10, 10) and (25, 25), "
            "coverage_radius = 8, "
            "automatic flight_buffer, cohere = 0.25, and match = 0.05."
        )

    profile_radius_pairs_are_valid = (
        (
            results_df["radius_profile"].eq("rd10_rp10_nominal")
            & results_df["drone_sensing_radius"].eq(10.0)
            & results_df["point_sensing_radius"].eq(10.0)
        )
        |
        (
            results_df["radius_profile"].eq("rd25_rp25_potenziato")
            & results_df["drone_sensing_radius"].eq(25.0)
            & results_df["point_sensing_radius"].eq(25.0)
        )
    ).all()

    if not profile_radius_pairs_are_valid:
        raise RuntimeError("E26b radius-profile labels do not match their radii.")

    configuration_columns = [
        "radius_profile",
        "drone_sensing_radius",
        "point_sensing_radius",
        "point_routine",
    ]
    actual_configurations = (
        results_df[configuration_columns]
        .drop_duplicates()
        .shape[0]
    )

    if actual_configurations != expected_configurations:
        raise RuntimeError(
            f"Expected {expected_configurations} configurations, but found "
            f"{actual_configurations}."
        )

    runs_per_configuration = results_df.groupby(
        configuration_columns
    )["RunId"].nunique()

    if not runs_per_configuration.eq(len(seeds)).all():
        raise RuntimeError(
            "The E26b configuration contains an unexpected number of runs."
        )

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

    # Create the experiment directory only after the batch has completed and its results have passed validation.
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
        "excess_service_units",
    ]

    print(f"Collected rows: {len(results_df)}")
    print(results_df[columns_to_show])

if __name__ == "__main__":
    main()
