"""Analyze saved batch results without rerunning the simulation."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

EXPERIMENT_NAME = "coverage_radius_static_pilot"

# Model parameters whose values distinguish the configurations being compared.
COMPARISON_PARAMETERS = ["coverage_radius"]
# Fraction of nominally obtainable service required by the response-time metric.
COVERAGE_THRESHOLD = 0.90

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIRECTORY = PROJECT_ROOT / "results" / EXPERIMENT_NAME

INPUT_PATH = EXPERIMENT_DIRECTORY / "raw_results.csv"
CONFIG_PATH = EXPERIMENT_DIRECTORY / "experiment_config.json"

SUMMARY_DIRECTORY = EXPERIMENT_DIRECTORY / "summaries"
SUMMARY_PATH = SUMMARY_DIRECTORY / "run_summary.csv"
AGGREGATE_PATH = SUMMARY_DIRECTORY / "aggregate_summary.csv"
TIME_SERIES_PATH = SUMMARY_DIRECTORY / "time_series_summary.csv"
EVENT_RESPONSE_PATH = (SUMMARY_DIRECTORY / "event_response_summary.csv")

FIGURES_DIRECTORY = EXPERIMENT_DIRECTORY / "figures"
DEFICIT_FIGURE_PATH = FIGURES_DIRECTORY / "normalized_deficit.png"
R_DELTA_FIGURE_PATH = FIGURES_DIRECTORY / "deficit_reduction.png"
FLEET_STATE_FIGURE_PATH = FIGURES_DIRECTORY / "fleet_state.png"
J_DELTA_COMPARISON_FIGURE_PATH = FIGURES_DIRECTORY / "j_delta_comparison.png"
POINT_STATE_FIGURE_PATH = FIGURES_DIRECTORY / "point_service_state.png"
CAPACITY_ADJUSTED_COVERAGE_FIGURE_PATH = (FIGURES_DIRECTORY / "capacity_adjusted_coverage.png")
TIME_TO_90_PERCENT_FIGURE_PATH = (FIGURES_DIRECTORY / "time_to_90_percent_nominal_service.png")
SCENARIO_CHARACTERISTICS_FIGURE_PATH = (FIGURES_DIRECTORY / "scenario_characteristics.png")
EVENT_RESPONSE_FIGURE_PATH = (FIGURES_DIRECTORY / "event_response_time.png")


def load_event_markers(config_path):
    """Load and validate the high-level event markers saved for an experiment."""
    with config_path.open("r", encoding="utf-8") as config_file:
        experiment_config = json.load(config_file)

    event_markers = experiment_config.get("event_markers")

    if not isinstance(event_markers, list):
        raise ValueError(
            "Experiment configuration does not contain a valid event_markers list."
        )

    required_fields = {"step", "simulated_time_s", "event_type"}

    for marker in event_markers:
        if not isinstance(marker, dict):
            raise ValueError("Each event marker must be a dictionary.")

        missing_fields = required_fields - set(marker)

        if missing_fields:
            raise ValueError(
                f"Event marker is missing fields: {sorted(missing_fields)}"
            )

    return sorted(
        event_markers,
        key=lambda marker: marker["simulated_time_s"],
    )

def build_event_episodes(event_markers):
    """Group simultaneous high-level events into observable event episodes."""
    episodes_by_step = {}

    for marker in event_markers:
        event_step = int(marker["step"])
        event_time_s = float(marker["simulated_time_s"])
        event_type = str(marker["event_type"])

        if event_step not in episodes_by_step:
            episodes_by_step[event_step] = {
                "event_time_s": event_time_s,
                "event_types": [],
            }
        elif episodes_by_step[event_step]["event_time_s"] != event_time_s:
            raise ValueError(
                f"Event markers at step {event_step} have inconsistent times."
            )

        if event_type not in episodes_by_step[event_step]["event_types"]:
            episodes_by_step[event_step]["event_types"].append(event_type)

    event_episodes = []

    for event_index, event_step in enumerate(
        sorted(episodes_by_step),
        start=1,
    ):
        episode = episodes_by_step[event_step]

        event_episodes.append(
            {
                "event_index": event_index,
                "event_step": event_step,
                "event_time_s": episode["event_time_s"],
                "event_types": " + ".join(episode["event_types"]),
            }
        )

    return event_episodes


def get_configuration_groups(data, comparison_parameters):
    """Return one labeled data subset for each configuration."""
    if not comparison_parameters:
        return [("Mean across replications", data)]

    if len(comparison_parameters) == 1:
        grouping_key = comparison_parameters[0]
    else:
        grouping_key = comparison_parameters

    configuration_groups = []

    for values, configuration_df in data.groupby(
        grouping_key,
        sort=True,
        dropna=False
    ):
        if len(comparison_parameters) == 1:
            values = (values,)

        label_parts = []

        for parameter, value in zip(comparison_parameters, values):
            label_part = f"{parameter}={value}"
            label_parts.append(label_part)

        label = ", ".join(label_parts)

        configuration_groups.append((label, configuration_df))

    return configuration_groups


def add_event_markers(axis, event_markers):
    """Draw one dashed vertical line for each observable event episode."""
    event_episodes = build_event_episodes(event_markers)

    for episode in event_episodes:
        event_time_s = episode["event_time_s"]
        event_label = (
            episode["event_types"]
            .replace("_", " ")
            .title()
        )

        axis.axvline(
            x=event_time_s,
            color="black",
            linestyle="--",
            label=f"Event: {event_label} (t={event_time_s:g} s)",
        )


def calculate_event_response_time_s(
    run_data,
    event_step,
    event_time_s,
    next_event_step,
    threshold,
):
    """Return the time needed to reach the threshold within one event window."""
    event_window = run_data[run_data["Step"] >= event_step]

    if next_event_step is not None:
        event_window = event_window[event_window["Step"] < next_event_step]

    threshold_rows = event_window[event_window["capacity_adjusted_coverage"] >= threshold]

    if threshold_rows.empty:
        return float("nan")

    first_threshold_time_s = threshold_rows["simulated_time_s"].min()
    response_time_s = float(first_threshold_time_s - event_time_s)

    if response_time_s < -1e-9:
        raise ValueError(
            "Threshold time precedes the corresponding event time."
        )

    return max(0.0, response_time_s)


def build_event_response_summary(
    data,
    comparison_parameters,
    event_markers,
    threshold,
):
    """Return one response-time row for each run and event episode."""
    event_episodes = build_event_episodes(event_markers)

    output_columns = [
        "RunId",
        "seed",
        *comparison_parameters,
        "event_index",
        "event_step",
        "event_time_s",
        "event_types",
        "next_event_step",
        "coverage_threshold",
        "threshold_reached",
        "time_to_90_percent_nominal_service_after_event_s",
    ]

    response_records = []

    for run_id, run_data in data.groupby("RunId", sort=True):
        run_data = run_data.sort_values("Step")
        last_observed_step = int(run_data["Step"].max())

        for episode_position, episode in enumerate(event_episodes):
            event_step = episode["event_step"]

            if event_step > last_observed_step:
                raise ValueError(
                    f"Event at step {event_step} lies beyond run "
                    f"{run_id}, which ends at step {last_observed_step}."
                )

            if episode_position + 1 < len(event_episodes):
                next_event_step = event_episodes[
                    episode_position + 1
                ]["event_step"]
            else:
                next_event_step = None

            response_time_s = calculate_event_response_time_s(
                run_data=run_data,
                event_step=event_step,
                event_time_s=episode["event_time_s"],
                next_event_step=next_event_step,
                threshold=threshold,
            )

            response_record = {
                "RunId": run_id,
                "seed": run_data["seed"].iloc[0],
            }

            for parameter in comparison_parameters:
                response_record[parameter] = run_data[parameter].iloc[0]

            response_record.update(
                {
                    "event_index": episode["event_index"],
                    "event_step": event_step,
                    "event_time_s": episode["event_time_s"],
                    "event_types": episode["event_types"],
                    "next_event_step": next_event_step,
                    "coverage_threshold": threshold,
                    "threshold_reached": not pd.isna(response_time_s),
                    "time_to_90_percent_nominal_service_after_event_s": (
                        response_time_s
                    ),
                }
            )

            response_records.append(response_record)

    return pd.DataFrame.from_records(
        response_records,
        columns=output_columns,
    )


def save_time_series_plot(
    data,
    comparison_parameters,
    mean_column,
    std_column,
    output_path,
    title,
    y_label,
    y_limits=None,
    reference_y=None,
    reference_label=None,
    reference_series_column=None,
    reference_series_label=None,
    event_markers=(),
):
    """Save a time-series mean with a one-standard-deviation band."""
    figure, axis = plt.subplots(figsize=(8, 5))

    configuration_groups = get_configuration_groups(data, comparison_parameters)

    for configuration_label, configuration_df in configuration_groups:
        time_s = configuration_df["simulated_time_s"]
        mean_values = configuration_df[mean_column]

        # With one replication the sample standard deviation is undefined.
        # In that case, draw a zero-width variability band.
        std_values = configuration_df[std_column].fillna(0.0)

        if comparison_parameters:
            line_label = (f"{configuration_label} (mean ± 1 SD)")
            band_label = None
        else:
            line_label = configuration_label
            band_label = "Mean ± 1 standard deviation"

        # plot() returns a list; this call creates exactly one line.
        line = axis.plot(time_s, mean_values, label=line_label)[0]

        axis.fill_between(
            time_s,
            mean_values - std_values,
            mean_values + std_values,
            color=line.get_color(),
            alpha=0.2,
            label=band_label,
        )

        if reference_series_column is not None:
            if comparison_parameters:
                reference_curve_label = (f"{configuration_label}: {reference_series_label}")
            else:
                reference_curve_label = reference_series_label

            axis.plot(
                time_s,
                configuration_df[reference_series_column],
                color=line.get_color(),
                linestyle="--",
                linewidth=1.2,
                label=reference_curve_label,
            )

    axis.set_xlabel("Simulated time (s; 1 step = 1 s)")
    all_time_s = data["simulated_time_s"]
    axis.set_xlim(all_time_s.min(), all_time_s.max())
    axis.set_ylabel(y_label)
    axis.set_title(title)

    if y_limits is not None:
        axis.set_ylim(*y_limits)

    if reference_y is not None:
        axis.axhline(
            reference_y,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=reference_label,
        )

    add_event_markers(axis, event_markers)

    axis.grid(alpha=0.3)
    axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)



def save_j_delta_comparison_plot(
    data,
    comparison_parameters,
    output_path,
):
    """Save mean J_delta with one-standard-deviation error bars."""
    if len(comparison_parameters) != 1:
        raise ValueError("The J_delta comparison plot requires exactly one comparison parameter.")

    parameter = comparison_parameters[0]
    plot_data = data.sort_values(parameter)

    x_values = plot_data[parameter]
    mean_values = plot_data["J_delta_mean"]
    std_values = plot_data["J_delta_std"].fillna(0.0)

    # Connect ordered numeric values, but do not imply continuity between categorical parameter values.
    if pd.api.types.is_numeric_dtype(x_values):
        plot_format = "o-"
    else:
        plot_format = "o"

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.errorbar(
        x_values,
        mean_values,
        yerr=std_values,
        fmt=plot_format,
        capsize=4,
        label="Mean ± 1 standard deviation",
    )

    axis.set_xlabel(parameter)
    axis.set_ylabel(r"Mean $J_\Delta$")
    axis.set_ylim(bottom=0.0)
    axis.set_title(r"Aggregate normalized deficit ($J_\Delta$)")
    axis.grid(alpha=0.3)
    axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_time_to_90_percent_comparison_plot(
    data,
    comparison_parameters,
    output_path,
):
    """Save threshold-time statistics for each compared configuration."""
    if len(comparison_parameters) != 1:
        raise ValueError(
            "The threshold-time comparison plot requires exactly "
            "one comparison parameter."
        )

    parameter = comparison_parameters[0]
    plot_data = data.sort_values(parameter).reset_index(drop=True)

    mean_values = plot_data["time_to_90_percent_nominal_service_mean_s"]
    std_values = plot_data["time_to_90_percent_nominal_service_std_s"].fillna(0.0)

    reached_mask = mean_values.notna()

    figure, axis = plt.subplots(figsize=(8, 5))

    if reached_mask.any():
        reached_positions = plot_data.index[reached_mask]

        axis.errorbar(
            reached_positions,
            mean_values[reached_mask],
            yerr=std_values[reached_mask],
            fmt="o",
            capsize=4,
            label="Mean ± 1 standard deviation",
        )

    for position, row in plot_data.iterrows():
        reached_runs = int(row["runs_reaching_90_percent_nominal_service"])
        total_runs = int(row["replications"])
        reached_label = f"{reached_runs}/{total_runs} reached"

        mean_time_s = row["time_to_90_percent_nominal_service_mean_s"]

        if pd.notna(mean_time_s):
            axis.annotate(
                reached_label,
                xy=(position, mean_time_s),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
            )
        else:
            axis.text(
                position,
                0.03,
                reached_label,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
            )
    
    axis.set_xticks(plot_data.index)
    axis.set_xticklabels(plot_data[parameter].astype(str))
    axis.set_xlabel(parameter)
    axis.set_ylabel("Mean threshold time among reached runs (s)")
    axis.margins(y=0.15)
    axis.set_ylim(bottom=0.0)
    axis.set_title(f"Time to {COVERAGE_THRESHOLD:.0%} of nominally obtainable service")
    axis.grid(alpha=0.3)

    if reached_mask.any():
        axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_event_response_comparison_plot(
    data,
    comparison_parameters,
    output_path,
):
    """Save mean event-response statistics for each configuration."""
    if len(comparison_parameters) != 1:
        raise ValueError(
            "The event-response comparison plot requires exactly "
            "one comparison parameter."
        )

    parameter = comparison_parameters[0]
    plot_data = data.sort_values(parameter).reset_index(drop=True)

    mean_values = plot_data["mean_event_response_time_mean_s"]
    std_values = (
        plot_data["mean_event_response_time_std_s"]
        .fillna(0.0)
    )

    reached_mask = mean_values.notna()

    figure, axis = plt.subplots(figsize=(8, 5))

    if reached_mask.any():
        reached_positions = plot_data.index[reached_mask]

        axis.errorbar(
            reached_positions,
            mean_values[reached_mask],
            yerr=std_values[reached_mask],
            fmt="o",
            capsize=4,
            label="Mean across runs ± 1 standard deviation",
        )

    for position, row in plot_data.iterrows():
        reached_events = int(
            row[
                "event_episode_cases_reaching_90_percent_nominal_service"
            ]
        )
        total_events = int(row["event_episode_cases"])
        contributing_runs = int(
            row["runs_contributing_to_event_response_mean"]
        )
        total_runs = int(row["replications"])

        reached_label = (
            f"{reached_events}/{total_events} events; "
            f"{contributing_runs}/{total_runs} runs"
        )

        mean_time_s = row["mean_event_response_time_mean_s"]

        if pd.notna(mean_time_s):
            axis.annotate(
                reached_label,
                xy=(position, mean_time_s),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
            )
        else:
            axis.text(
                position,
                0.03,
                reached_label,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
            )

    axis.set_xticks(plot_data.index)
    axis.set_xticklabels(plot_data[parameter].astype(str))
    axis.set_xlabel(parameter)
    axis.set_ylabel("Mean per-run event response time (s)")
    axis.margins(y=0.15)
    axis.set_ylim(bottom=0.0)
    axis.set_title(
        f"Response time to {COVERAGE_THRESHOLD:.0%} "
        "of nominally obtainable service after events"
    )
    axis.grid(alpha=0.3)

    if reached_mask.any():
        axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_fleet_state_plot(
    data,
    comparison_parameters,
    output_path,
    event_markers=()
):
    """Save fleet-state time series for each configuration."""
    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(8, 10),
        sharex=True,
        sharey=True
    )

    series = (
        (
            "idle_drones_mean",
            "idle_drones_std",
            "Idle drones",
        ),
        (
            "exploring_drones_mean",
            "exploring_drones_std",
            "Exploring drones",
        ),
        (
            "stationing_drones_mean",
            "stationing_drones_std",
            "Stationing drones",
        ),
    )

    configuration_groups = get_configuration_groups(
        data,
        comparison_parameters
    )

    for axis, (mean_column, std_column, state_label) in zip(axes, series):
        for configuration_label, configuration_df in configuration_groups:
            time_s = configuration_df["simulated_time_s"]
            mean_values = configuration_df[mean_column]
            std_values = configuration_df[std_column].fillna(0.0)

            if comparison_parameters:
                line_label = (
                    f"{configuration_label} (mean ± 1 SD)"
                )
            else:
                line_label = "Mean ± 1 standard deviation"

            line = axis.plot(
                time_s,
                mean_values,
                label=line_label
            )[0]

            axis.fill_between(
                time_s,
                mean_values - std_values,
                mean_values + std_values,
                color=line.get_color(),
                alpha=0.15
            )

        axis.set_ylabel("Number of drones")
        axis.set_ylim(bottom=0.0)
        axis.set_title(state_label)
        add_event_markers(axis, event_markers)
        axis.grid(alpha=0.3)
        axis.legend()

    all_time_s = data["simulated_time_s"]
    axes[-1].set_xlim(all_time_s.min(), all_time_s.max())
    axes[-1].set_xlabel("Simulated time (s; 1 step = 1 s)")

    figure.suptitle("Fleet state over time")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_point_state_plot(
    data,
    comparison_parameters,
    output_path,
    event_markers=()
):
    """Save point-service-state time series for each configuration."""
    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(8, 10),
        sharex=True,
        sharey=True
    )

    series = (
        (
            "underserved_points_mean",
            "underserved_points_std",
            "Underserved points",
        ),
        (
            "exactly_satisfied_points_mean",
            "exactly_satisfied_points_std",
            "Exactly satisfied points",
        ),
        (
            "overserved_points_mean",
            "overserved_points_std",
            "Overserved points",
        ),
    )

    configuration_groups = get_configuration_groups(data, comparison_parameters)

    for axis, (mean_column, std_column, state_label) in zip(axes, series):
        for configuration_label, configuration_df in configuration_groups:
            time_s = configuration_df["simulated_time_s"]
            mean_values = configuration_df[mean_column]
            std_values = configuration_df[std_column].fillna(0.0)

            if comparison_parameters:
                line_label = (
                    f"{configuration_label} (mean ± 1 SD)"
                )
            else:
                line_label = "Mean ± 1 standard deviation"

            line = axis.plot(
                time_s,
                mean_values,
                label=line_label
            )[0]

            axis.fill_between(
                time_s,
                mean_values - std_values,
                mean_values + std_values,
                color=line.get_color(),
                alpha=0.15
            )

            if comparison_parameters:
                active_points_label = (f"{configuration_label}: active points")
            else:
                active_points_label = "Active points"

            axis.plot(
                time_s,
                configuration_df["active_points_mean"],
                color=line.get_color(),
                linestyle="--",
                linewidth=1.2,
                label=active_points_label,
            )

        axis.set_ylabel("Number of points")
        axis.set_ylim(bottom=0.0)
        axis.set_title(state_label)
        add_event_markers(axis, event_markers)
        axis.grid(alpha=0.3)
        axis.legend()

    all_time_s = data["simulated_time_s"]
    axes[-1].set_xlim(all_time_s.min(), all_time_s.max())
    axes[-1].set_xlabel("Simulated time (s; 1 step = 1 s)")

    figure.suptitle("Point service state over time")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

def save_scenario_characteristics_plot(
    data,
    comparison_parameters,
    output_path,
    event_markers=()
):
    """Save time-series plots of scenario characteristics."""
    figure, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(8, 15),
        sharex=True,
    )

    series = (
        (
            "active_points_mean",
            "active_points_std",
            "Active points",
            "Number of points",
        ),
        (
            "total_demand_mean",
            "total_demand_std",
            "Total demand",
            "Demand units",
        ),
        (
            "mean_demand_per_point_mean",
            "mean_demand_per_point_std",
            "Mean demand per point",
            "Demand units per point",
        ),
        (
            "fleet_load_mean",
            "fleet_load_std",
            "Fleet load",
            "Demand units per drone",
        ),
        (
            "overlapping_zones_mean",
            "overlapping_zones_std",
            "Overlapping coverage-zone pairs",
            "Number of point pairs",
        ),
    )

    configuration_groups = get_configuration_groups(
        data,
        comparison_parameters,
    )

    for axis, (
        mean_column,
        std_column,
        panel_title,
        y_label,
    ) in zip(axes, series):
        for configuration_label, configuration_df in configuration_groups:
            time_s = configuration_df["simulated_time_s"]
            mean_values = configuration_df[mean_column]
            std_values = configuration_df[std_column].fillna(0.0)

            if comparison_parameters:
                line_label = (
                    f"{configuration_label} (mean ± 1 SD)"
                )
            else:
                line_label = "Mean ± 1 standard deviation"

            line = axis.plot(
                time_s,
                mean_values,
                label=line_label,
            )[0]

            axis.fill_between(
                time_s,
                mean_values - std_values,
                mean_values + std_values,
                color=line.get_color(),
                alpha=0.15,
            )

        axis.set_ylabel(y_label)
        axis.set_ylim(bottom=0.0)
        axis.set_title(panel_title)
        add_event_markers(axis, event_markers)
        axis.grid(alpha=0.3)
        axis.legend()

    all_time_s = data["simulated_time_s"]
    axes[-1].set_xlim(all_time_s.min(), all_time_s.max())
    axes[-1].set_xlabel("Simulated time (s; 1 step = 1 s)")

    figure.suptitle("Scenario characteristics over time")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main():
    """Load and analyze saved batch results."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Batch result file not found: {INPUT_PATH}")

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Experiment configuration file not found: {CONFIG_PATH}"
        )

    event_markers = load_event_markers(CONFIG_PATH)

    results_df = pd.read_csv(INPUT_PATH)

    # Fail early if a requested comparison parameter is absent from the CSV.
    missing_comparison_parameters = (set(COMPARISON_PARAMETERS) - set(results_df.columns))

    if missing_comparison_parameters:
        raise ValueError("Comparison parameters are missing from the batch results: "f"{sorted(missing_comparison_parameters)}")

    # Derive metrics that do not require additional model reporters.
    useful_service = (results_df["total_demand"] - results_df["residual_deficit"])

    nominal_service_capacity = (results_df[["total_demand", "n_drones"]].min(axis=1))

    results_df["capacity_adjusted_coverage"] = (
        useful_service
        .div(nominal_service_capacity)
        .where(nominal_service_capacity > 0, 1.0)
        .clip(upper=1.0)
    )

    # Calculate one threshold-response time for each run and event episode.
    event_response_df = build_event_response_summary(
        data=results_df,
        comparison_parameters=COMPARISON_PARAMETERS,
        event_markers=event_markers,
        threshold=COVERAGE_THRESHOLD,
    )

    # Find the first simulated time at which each run reaches the threshold.
    threshold_reached_rows = results_df[results_df["capacity_adjusted_coverage"] >= COVERAGE_THRESHOLD]

    first_threshold_time_by_run_s = (
        threshold_reached_rows
        .groupby("RunId")["simulated_time_s"]
        .min()
    )

    results_df["mean_demand_per_point"] = (
        results_df["total_demand"]
        .div(results_df["active_points"])
        .where(results_df["active_points"] > 0)
    )

    results_df["fleet_load"] = (
        results_df["total_demand"]
        .div(results_df["n_drones"])
    )

    results_df["normalized_unavoidable_deficit"] = (
        results_df["unavoidable_deficit"]
        .div(results_df["total_demand"])
        .where(results_df["total_demand"] > 0, 0.0)
    )

    columns_to_show = [
        "RunId",
        "Step",
        "seed",
        "simulated_time_s",
        "residual_deficit",
        "normalized_deficit",
        "capacity_adjusted_coverage",
        "underserved_points",
        "exactly_satisfied_points",
        "overserved_points"
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

    # Keep comparison parameters attached to each summarized model run.
    run_group_columns = ["RunId"] + COMPARISON_PARAMETERS


    # Produce one summary row for each independent model execution.
    run_summary_df = (
        evaluation_rows
        .groupby(run_group_columns, as_index=False)
        .agg(
            seed=("seed", "first"),
            J_delta=("normalized_deficit", "mean"),
            final_time_s=("simulated_time_s", "last"),
            final_residual_deficit=("residual_deficit", "last"),
            final_normalized_deficit=("normalized_deficit", "last")
        )
    )

    run_summary_df["time_to_90_percent_nominal_service_s"] = (
        run_summary_df["RunId"]
        .map(first_threshold_time_by_run_s)
    )

    # Summarize event responses within each independent run.
    event_response_by_run_df = (
        event_response_df
        .groupby("RunId", as_index=False)
        .agg(
            event_episodes=("event_index", "size"),
            event_episodes_reaching_90_percent_nominal_service=(
                "time_to_90_percent_nominal_service_after_event_s",
                "count",
            ),
            mean_time_to_90_percent_nominal_service_after_event_s=(
                "time_to_90_percent_nominal_service_after_event_s",
                "mean",
            ),
        )
    )

    run_summary_df = run_summary_df.merge(
        event_response_by_run_df,
        on="RunId",
        how="left",
        validate="one_to_one",
    )

    event_count_columns = [
        "event_episodes",
        "event_episodes_reaching_90_percent_nominal_service",
    ]

    run_summary_df[event_count_columns] = (
        run_summary_df[event_count_columns]
        .fillna(0)
        .astype(int)
    )

    # Calculate r_delta(t) separately within each replication.
    results_df = results_df.sort_values(["RunId", "Step"]).copy()

    previous_normalized_deficit = (
        results_df
        .groupby("RunId")["normalized_deficit"]
        .shift(1)
    )

    results_df["r_delta"] = (previous_normalized_deficit - results_df["normalized_deficit"])

    # Average each step only across replications of the same configuration.
    time_series_group_columns = COMPARISON_PARAMETERS + ["Step"]

    # Aggregate the same simulation step across independent replications.
    # Step 0 is retained because it represents the initial state.
    time_series_summary_df = (
        results_df
        .groupby(time_series_group_columns, as_index=False)
        .agg(
            simulated_time_s=("simulated_time_s", "first"),
            replications=("RunId", "nunique"),

            residual_deficit_mean=("residual_deficit", "mean"),
            residual_deficit_std=("residual_deficit", "std"),
            normalized_deficit_mean=("normalized_deficit", "mean"),
            normalized_deficit_std=("normalized_deficit", "std"),
            capacity_adjusted_coverage_mean=("capacity_adjusted_coverage", "mean"),
            capacity_adjusted_coverage_std=("capacity_adjusted_coverage", "std"),
            normalized_unavoidable_deficit_mean=("normalized_unavoidable_deficit", "mean"),
            normalized_unavoidable_deficit_std=("normalized_unavoidable_deficit", "std"),
            r_delta_mean=("r_delta", "mean"),
            r_delta_std=("r_delta", "std"),

            underserved_points_mean=("underserved_points", "mean"),
            underserved_points_std=("underserved_points", "std"),
            exactly_satisfied_points_mean=("exactly_satisfied_points", "mean"),
            exactly_satisfied_points_std=("exactly_satisfied_points", "std"),
            overserved_points_mean=("overserved_points", "mean"),
            overserved_points_std=("overserved_points", "std"),

            idle_drones_mean=("idle_drones", "mean"),
            idle_drones_std=("idle_drones", "std"),
            exploring_drones_mean=("exploring_drones", "mean"),
            exploring_drones_std=("exploring_drones", "std"),
            stationing_drones_mean=("stationing_drones", "mean"),
            stationing_drones_std=("stationing_drones", "std"),

            active_points_mean=("active_points", "mean"),
            active_points_std=("active_points", "std"),
            total_demand_mean=("total_demand", "mean"),
            total_demand_std=("total_demand", "std"),
            mean_demand_per_point_mean=("mean_demand_per_point", "mean"),
            mean_demand_per_point_std=("mean_demand_per_point", "std"),
            fleet_load_mean=("fleet_load", "mean"),
            fleet_load_std=("fleet_load", "std"),
            overlapping_zones_mean=("overlapping_zones", "mean"),
            overlapping_zones_std=("overlapping_zones", "std")
        )
    )

    # Pandas requires at least one grouping column.
    # When no comparison parameter is selected,
    # a temporary constant identifies the single configuration without altering run_summary_df.
    if COMPARISON_PARAMETERS:
        aggregate_source_df = run_summary_df
        aggregate_group_columns = COMPARISON_PARAMETERS
    else:
        aggregate_source_df = run_summary_df.assign(
            _single_configuration=0
        )
        aggregate_group_columns = ["_single_configuration"]

    # Aggregate independent replications separately for each configuration.
    # Pandas "std" uses the sample standard deviation (ddof=1).
    aggregate_summary_df = (
        aggregate_source_df
        .groupby(aggregate_group_columns, as_index=False)
        .agg(
            replications=("RunId", "nunique"),
            J_delta_mean=("J_delta", "mean"),
            J_delta_std=("J_delta", "std"),
            runs_reaching_90_percent_nominal_service=("time_to_90_percent_nominal_service_s", "count"),
            time_to_90_percent_nominal_service_mean_s=("time_to_90_percent_nominal_service_s", "mean"),
            time_to_90_percent_nominal_service_std_s=("time_to_90_percent_nominal_service_s", "std"),
            event_episode_cases=("event_episodes", "sum"),
            event_episode_cases_reaching_90_percent_nominal_service=(
                "event_episodes_reaching_90_percent_nominal_service",
                "sum",
            ),
            runs_contributing_to_event_response_mean=(
                "mean_time_to_90_percent_nominal_service_after_event_s",
                "count",
            ),
            mean_event_response_time_mean_s=(
                "mean_time_to_90_percent_nominal_service_after_event_s",
                "mean",
            ),
            mean_event_response_time_std_s=(
                "mean_time_to_90_percent_nominal_service_after_event_s",
                "std",
            ),
            final_residual_deficit_mean=("final_residual_deficit", "mean"),
            final_residual_deficit_std=("final_residual_deficit", "std"),
            final_normalized_deficit_mean=("final_normalized_deficit", "mean"),
            final_normalized_deficit_std=("final_normalized_deficit", "std"),
        )
    )

    # Remove the temporary key from the saved single-configuration summary.
    if not COMPARISON_PARAMETERS:
        aggregate_summary_df = aggregate_summary_df.drop(
            columns="_single_configuration"
        )

    # Save one compact summary row for each model execution.
    SUMMARY_DIRECTORY.mkdir(parents=True, exist_ok=True)

    run_summary_df.to_csv(SUMMARY_PATH, index=False)
    aggregate_summary_df.to_csv(AGGREGATE_PATH, index=False)
    time_series_summary_df.to_csv(TIME_SERIES_PATH, index=False)
    event_response_df.to_csv(EVENT_RESPONSE_PATH, index=False)

    save_time_series_plot(
        data=time_series_summary_df,
        comparison_parameters=COMPARISON_PARAMETERS,
        mean_column="normalized_deficit_mean",
        std_column="normalized_deficit_std",
        output_path=DEFICIT_FIGURE_PATH,
        title="Normalized deficit over time",
        y_label="Fraction of total demand unmet",
        y_limits=(0.0, 1.05),
        reference_series_column="normalized_unavoidable_deficit_mean",
        reference_series_label="Mean conditional structural reference",
        event_markers=event_markers
    )

    save_time_series_plot(
        data=time_series_summary_df,
        comparison_parameters=COMPARISON_PARAMETERS,
        mean_column="capacity_adjusted_coverage_mean",
        std_column="capacity_adjusted_coverage_std",
        output_path=CAPACITY_ADJUSTED_COVERAGE_FIGURE_PATH,
        title="Capacity-adjusted coverage over time",
        y_label="Fraction of nominally obtainable service",
        y_limits=(0.0, 1.05),
        reference_y=COVERAGE_THRESHOLD,
        reference_label=f"{COVERAGE_THRESHOLD:.0%} threshold",
        event_markers=event_markers,
    )

    save_time_series_plot(
        data=time_series_summary_df,
        comparison_parameters=COMPARISON_PARAMETERS,
        mean_column="r_delta_mean",
        std_column="r_delta_std",
        output_path=R_DELTA_FIGURE_PATH,
        title="Normalized deficit reduction over time",
        y_label="Normalized deficit reduction per step",
        reference_y=0.0,
        reference_label="No change",
        event_markers=event_markers
    )

    save_point_state_plot(
        data=time_series_summary_df,
        comparison_parameters=COMPARISON_PARAMETERS,
        output_path=POINT_STATE_FIGURE_PATH,
        event_markers=event_markers,
    )

    save_fleet_state_plot(
        data=time_series_summary_df,
        comparison_parameters=COMPARISON_PARAMETERS,
        output_path=FLEET_STATE_FIGURE_PATH,
        event_markers=event_markers
    )

    save_scenario_characteristics_plot(
        data=time_series_summary_df,
        comparison_parameters=COMPARISON_PARAMETERS,
        output_path=SCENARIO_CHARACTERISTICS_FIGURE_PATH,
        event_markers=event_markers
    )

    if len(COMPARISON_PARAMETERS) == 1:
        save_j_delta_comparison_plot(
            data=aggregate_summary_df,
            comparison_parameters=COMPARISON_PARAMETERS,
            output_path=J_DELTA_COMPARISON_FIGURE_PATH,
        )

        save_time_to_90_percent_comparison_plot(
            data=aggregate_summary_df,
            comparison_parameters=COMPARISON_PARAMETERS,
            output_path=TIME_TO_90_PERCENT_FIGURE_PATH,
        )

        if not event_response_df.empty:
            save_event_response_comparison_plot(
                data=aggregate_summary_df,
                comparison_parameters=COMPARISON_PARAMETERS,
                output_path=EVENT_RESPONSE_FIGURE_PATH,
            )

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

    print()
    print("Event-response summary:")
    print(event_response_df)
    print(f"Event-response summary saved to: {EVENT_RESPONSE_PATH}")

    print()
    print(f"Deficit figure saved to: {DEFICIT_FIGURE_PATH}")
    print(f"Capacity-adjusted coverage figure saved to: "f"{CAPACITY_ADJUSTED_COVERAGE_FIGURE_PATH}")
    print(f"Deficit-reduction figure saved to: "f"{R_DELTA_FIGURE_PATH}")
    print(f"Fleet-state figure saved to: {FLEET_STATE_FIGURE_PATH}")
    print(f"Point-service-state figure saved to: {POINT_STATE_FIGURE_PATH}")
    print(f"Scenario characteristics figure saved to: "f"{SCENARIO_CHARACTERISTICS_FIGURE_PATH}")
    if len(COMPARISON_PARAMETERS) == 1:
        print("J_delta comparison figure saved to: "f"{J_DELTA_COMPARISON_FIGURE_PATH}")
        print(f"Time-to-threshold comparison figure saved to: "f"{TIME_TO_90_PERCENT_FIGURE_PATH}")
        if not event_response_df.empty:
            print("Event-response figure saved to: "f"{EVENT_RESPONSE_FIGURE_PATH}")

if __name__ == "__main__":
    main()