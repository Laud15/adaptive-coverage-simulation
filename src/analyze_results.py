"""Analyze saved batch results without rerunning the simulation."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

EXPERIMENT_NAME = "coverage_radius_static_pilot"

# Model parameters whose values distinguish the configurations being compared.
COMPARISON_PARAMETERS = ["coverage_radius"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIRECTORY = PROJECT_ROOT / "results" / EXPERIMENT_NAME

INPUT_PATH = EXPERIMENT_DIRECTORY / "raw_results.csv"

SUMMARY_DIRECTORY = EXPERIMENT_DIRECTORY / "summaries"
SUMMARY_PATH = SUMMARY_DIRECTORY / "run_summary.csv"
AGGREGATE_PATH = SUMMARY_DIRECTORY / "aggregate_summary.csv"
TIME_SERIES_PATH = SUMMARY_DIRECTORY / "time_series_summary.csv"

FIGURES_DIRECTORY = EXPERIMENT_DIRECTORY / "figures"
DEFICIT_FIGURE_PATH = FIGURES_DIRECTORY / "normalized_deficit.png"
R_DELTA_FIGURE_PATH = FIGURES_DIRECTORY / "deficit_reduction.png"
SATISFIED_FRACTION_FIGURE_PATH = FIGURES_DIRECTORY / "satisfied_fraction.png"
OVERSERVICE_FIGURE_PATH = FIGURES_DIRECTORY / "overservice.png"
FLEET_STATE_FIGURE_PATH = FIGURES_DIRECTORY / "fleet_state.png"
J_DELTA_COMPARISON_FIGURE_PATH = FIGURES_DIRECTORY / "j_delta_comparison.png"

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
    reference_label=None
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

    axis.set_xlabel("Simulated time (s)")
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



def save_fleet_state_plot(
    data,
    comparison_parameters,
    output_path
):
    """Save fleet-state time series for each configuration."""
    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 8),
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
        )
    )

    configuration_groups = get_configuration_groups(
        data,
        comparison_parameters
    )

    for axis, (
        mean_column,
        std_column,
        state_label
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
        axis.grid(alpha=0.3)
        axis.legend()

    all_time_s = data["simulated_time_s"]
    axes[-1].set_xlim(all_time_s.min(), all_time_s.max())
    axes[-1].set_xlabel("Simulated time (s)")

    figure.suptitle("Fleet state over time")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main():
    """Load and analyze saved batch results."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Batch result file not found: {INPUT_PATH}")

    results_df = pd.read_csv(INPUT_PATH)

    # Fail early if a requested comparison parameter is absent from the CSV.
    missing_comparison_parameters = (set(COMPARISON_PARAMETERS) - set(results_df.columns))

    if missing_comparison_parameters:
        raise ValueError("Comparison parameters are missing from the batch results: "f"{sorted(missing_comparison_parameters)}")

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
            final_residual_deficit_mean=(
                "final_residual_deficit",
                "mean",
            ),
            final_residual_deficit_std=(
                "final_residual_deficit",
                "std",
            ),
            final_normalized_deficit_mean=(
                "final_normalized_deficit",
                "mean",
            ),
            final_normalized_deficit_std=(
                "final_normalized_deficit",
                "std",
            ),
            final_satisfied_fraction_mean=(
                "final_satisfied_fraction",
                "mean",
            ),
            final_satisfied_fraction_std=(
                "final_satisfied_fraction",
                "std",
            ),
            final_overservice_mean=("final_overservice", "mean"),
            final_overservice_std=("final_overservice", "std"),
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


    save_time_series_plot(
        data=time_series_summary_df,
        comparison_parameters=COMPARISON_PARAMETERS,
        mean_column="normalized_deficit_mean",
        std_column="normalized_deficit_std",
        output_path=DEFICIT_FIGURE_PATH,
        title="Normalized deficit over time",
        y_label="Fraction of total demand unmet",
        y_limits=(0.0, 1.05)
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
        reference_label="No change"
    )

    save_time_series_plot(
        data=time_series_summary_df,
        comparison_parameters=COMPARISON_PARAMETERS,
        mean_column="satisfied_fraction_mean",
        std_column="satisfied_fraction_std",
        output_path=SATISFIED_FRACTION_FIGURE_PATH,
        title="Fraction of satisfied points over time",
        y_label="Fraction of active points satisfied",
        y_limits=(-0.05, 1.05)
    )

    save_time_series_plot(
        data=time_series_summary_df,
        comparison_parameters=COMPARISON_PARAMETERS,
        mean_column="overservice_mean",
        std_column="overservice_std",
        output_path=OVERSERVICE_FIGURE_PATH,
        title="Total overservice over time",
        y_label="Excess occupancy units across active points",
        reference_y=0.0,
        reference_label="No overservice"
    )

    save_fleet_state_plot(
        data=time_series_summary_df,
        comparison_parameters=COMPARISON_PARAMETERS,
        output_path=FLEET_STATE_FIGURE_PATH,
    )

    if len(COMPARISON_PARAMETERS) == 1:
        save_j_delta_comparison_plot(
            data=aggregate_summary_df,
            comparison_parameters=COMPARISON_PARAMETERS,
            output_path=J_DELTA_COMPARISON_FIGURE_PATH,
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
    print(f"Deficit figure saved to: {DEFICIT_FIGURE_PATH}")
    print(f"Deficit-reduction figure saved to: "f"{R_DELTA_FIGURE_PATH}")
    print(f"Satisfied-points figure saved to: "f"{SATISFIED_FRACTION_FIGURE_PATH}")
    print(f"Overservice figure saved to: {OVERSERVICE_FIGURE_PATH}")
    print(f"Fleet-state figure saved to: {FLEET_STATE_FIGURE_PATH}")
    if len(COMPARISON_PARAMETERS) == 1:
        print("J_delta comparison figure saved to: "f"{J_DELTA_COMPARISON_FIGURE_PATH}")

if __name__ == "__main__":
    main()