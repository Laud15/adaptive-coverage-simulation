import numpy as np
from PointsLayout import POINT_LAYOUTS, generate_point_positions

ROUTINE_EVENT_TYPES = (
    "reconfigure",
    "add_points",
    "remove_points",
    "change_priority_range",
    "set_priorities",
)

# Named technical routines shared by the interactive app and batch tests.
# These routines demonstrate the event mechanism and do not yet represent an application-specific phenomenon.
POINT_ROUTINES = {
    "static": (),
    "dynamic_demand_increase": (
        {
            "step": 600,
            "type": "change_priority_range",
            "min_priority": 2,
            "max_priority": 2,
        },
    ),
    "dynamic_demand_decrease": (
        {
            "step": 600,
            "type": "change_priority_range",
            "min_priority": 1,
            "max_priority": 1,
        },
    ),
    "dynamic_layout_random_to_circle": (
        {
            "step": 600,
            "type": "reconfigure",
            "layout": "circle",
            "n_points": 10,
        },
    ),
    "dynamic_layout_random_to_clusters": (
        {
            "step": 600,
            "type": "reconfigure",
            "layout": "clusters",
            "n_points": 10,
        },
    ),
    "dynamic_priority_redistribution": (
        {
            "step": 600,
            "type": "set_priorities",
            "priorities": (1, 3, 1, 3, 1, 3, 1, 3, 1, 3),
        },
    ),
    "dynamic_point_birth": (
        {
            "step": 600,
            "type": "add_points",
            "layout": "random",
            "n_points": 5,
        },
    ),
    "dynamic_point_death": (
        {
            "step": 600,
            "type": "remove_points",
            "point_idxs": (5, 6, 7, 8, 9),
        },
    ),
    "dynamic_composite_stress": (
        {
            "step": 600,
            "type": "set_priorities",
            "priorities": (1, 3, 1, 3, 1, 3, 1, 3, 1, 3),
        },
        {
            "step": 1200,
            "type": "reconfigure",
            "layout": "circle",
            "n_points": 10,
        },
        {
            "step": 1200,
            "type": "set_priorities",
            "priorities": (1, 3, 1, 3, 1, 3, 1, 3, 1, 3),
        },
        {
            "step": 1800,
            "type": "add_points",
            "layout": "random",
            "n_points": 5,
        },
        {
            "step": 1800,
            "type": "set_priorities",
            "priorities": (
                1, 3, 1, 3, 1, 3, 1, 3, 1, 3,
                2, 2, 2, 2, 2,
            ),
        },
        {
            "step": 2400,
            "type": "remove_points",
            "point_idxs": (20, 21, 22, 23, 24),
        },
        {
            "step": 2400,
            "type": "set_priorities",
            "priorities": (2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
        },
    ),
    "dynamic_composite_stress_300": (
        {
            "step": 600,
            "type": "set_priorities",
            "priorities": (1, 3, 1, 3, 1, 3, 1, 3, 1, 3),
        },
        {
            "step": 900,
            "type": "reconfigure",
            "layout": "circle",
            "n_points": 10,
        },
        {
            "step": 900,
            "type": "set_priorities",
            "priorities": (1, 3, 1, 3, 1, 3, 1, 3, 1, 3),
        },
        {
            "step": 1200,
            "type": "add_points",
            "layout": "random",
            "n_points": 5,
        },
        {
            "step": 1200,
            "type": "set_priorities",
            "priorities": (
                1, 3, 1, 3, 1, 3, 1, 3, 1, 3,
                2, 2, 2, 2, 2,
            ),
        },
        {
            "step": 1500,
            "type": "remove_points",
            "point_idxs": (20, 21, 22, 23, 24),
        },
        {
            "step": 1500,
            "type": "set_priorities",
            "priorities": (2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
        },
    ),
    "dynamic_composite_stress_150": (
        {
            "step": 600,
            "type": "set_priorities",
            "priorities": (1, 3, 1, 3, 1, 3, 1, 3, 1, 3),
        },
        {
            "step": 750,
            "type": "reconfigure",
            "layout": "circle",
            "n_points": 10,
        },
        {
            "step": 750,
            "type": "set_priorities",
            "priorities": (1, 3, 1, 3, 1, 3, 1, 3, 1, 3),
        },
        {
            "step": 900,
            "type": "add_points",
            "layout": "random",
            "n_points": 5,
        },
        {
            "step": 900,
            "type": "set_priorities",
            "priorities": (
                1, 3, 1, 3, 1, 3, 1, 3, 1, 3,
                2, 2, 2, 2, 2,
            ),
        },
        {
            "step": 1050,
            "type": "remove_points",
            "point_idxs": (20, 21, 22, 23, 24),
        },
        {
            "step": 1050,
            "type": "set_priorities",
            "priorities": (2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
        },
    ),
    "dynamic_demo": (
        {
            "step": 30,
            "type": "reconfigure",
            "layout": "dispersed",
            "n_points": 15,
        },
        {
            "step": 60,
            "type": "change_priority_range",
            "min_priority": 2,
            "max_priority": 4,
        },
        {
            "step": 90,
            "type": "reconfigure",
            "layout": "edges",
            "n_points": 8,
        },
    ),
}

def get_point_routine(name):
    """Return a named high-level point routine."""
    if not isinstance(name, str):
        raise TypeError("Point routine name must be a string.")

    try:
        return POINT_ROUTINES[name]
    except KeyError:
        raise ValueError(f"Unknown point routine {name!r}: "f"use one of {tuple(POINT_ROUTINES)}.") from None

def _require_integer(value, *, name, minimum):
    """Validate and return an integer routine parameter."""
    if isinstance(value, bool) or not isinstance(value,(int, np.integer)):
        raise TypeError(f"{name} must be an integer.")

    value = int(value)

    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")

    return value

def build_point_events(
    *,
    routine,
    initial_n_points,
    initial_min_priority,
    initial_max_priority,
    study_x_min,
    study_x_max,
    study_y_min,
    study_y_max,
    point_margin,
    event_seed
):
    """Compile a high-level point routine into model point events.

    Reconfiguration events remove every active point and create a new
    configuration. Add-points events preserve the existing points and create
    further ones; remove-points events delete an explicit subset. Priority-range
    events draw a new integer quota for every active point. Explicit-priority
    events assign a reproducible quota vector. Both priority event types update
    the range used by subsequent births.
    """
    if routine is None:
        return []

    if not isinstance(routine, (list, tuple)):
        raise TypeError("Point routine must be a list or tuple of events.")

    initial_n_points = _require_integer(initial_n_points, name="initial_n_points", minimum=0)
    initial_min_priority = _require_integer(initial_min_priority, name="initial_min_priority", minimum=1)
    initial_max_priority = _require_integer(initial_max_priority, name="initial_max_priority", minimum=1)
    event_seed = _require_integer(event_seed, name="event_seed", minimum=0)

    if initial_min_priority > initial_max_priority:
        raise ValueError(
            "initial_min_priority cannot be greater than "
            "initial_max_priority."
        )

    # This RNG is independent from the one used by the drone policy.
    event_rng = np.random.default_rng(event_seed)

    # Initial points receive identifiers from 0 to initial_n_points - 1.
    active_point_idxs = list(range(initial_n_points))
    next_point_idx = initial_n_points

    # Reconfigurations inherit the initial inclusive quota interval until a priority-range event changes it.
    current_priority_min = initial_min_priority
    current_priority_max = initial_max_priority

    point_events = []

    # Active identifiers are reconstructed in routine order, so events must be provided chronologically.
    # Equal steps remain allowed and preserve their written order.
    previous_step = 0

    for routine_position, routine_event in enumerate(routine):
        if not isinstance(routine_event, dict):
            raise TypeError("Each routine event must be represented by a dictionary.")

        if "step" not in routine_event or "type" not in routine_event:
            raise ValueError("Each routine event must contain 'step' and 'type'.")

        step = _require_integer(routine_event["step"], name=f"routine event {routine_position} step", minimum=1)

        if step < previous_step:
            raise ValueError(f"Routine event {routine_position} is scheduled at "f"step {step}, before the previous event at "f"step {previous_step}.")

        previous_step = step
        event_type = routine_event["type"]

        if (not isinstance(event_type, str) or event_type not in ROUTINE_EVENT_TYPES):
            raise ValueError(
                "Routine event type must be 'reconfigure', "
                "'add_points', 'remove_points', "
                "'change_priority_range', or 'set_priorities'."
            )

        if event_type == "reconfigure":
            if ("layout" not in routine_event or "n_points" not in routine_event):
                raise ValueError("A reconfiguration event must contain 'layout' and 'n_points'.")

            layout = routine_event["layout"]

            if layout not in POINT_LAYOUTS:
                raise ValueError(f"Unknown point layout {layout!r}: "f" use one of {POINT_LAYOUTS}.")

            n_points = _require_integer(routine_event["n_points"], name=f"reconfiguration at step {step} n_points", minimum=0)

            positions = generate_point_positions(
                layout=layout,
                n_points=n_points,
                study_x_min=study_x_min,
                study_x_max=study_x_max,
                study_y_min=study_y_min,
                study_y_max=study_y_max,
                margin=point_margin,
                rng=event_rng,
            )

            # Removals are emitted first so the event represents a complete replacement of the active point configuration.
            for point_idx in active_point_idxs:
                point_events.append(
                    {
                        "step": step,
                        "action": "remove",
                        "point_idx": point_idx,
                    }
                )

            new_active_point_idxs = []

            for position in positions:
                point_idx = next_point_idx
                next_point_idx += 1

                priority = int(event_rng.integers(current_priority_min, current_priority_max + 1))

                point_events.append(
                    {
                        "step": step,
                        "action": "create",
                        "point_idx": point_idx,
                        "position": (
                            float(position[0]),
                            float(position[1]),
                        ),
                        "priority": priority,
                    }
                )

                new_active_point_idxs.append(point_idx)

            active_point_idxs = new_active_point_idxs

        elif event_type == "add_points":
            if ("layout" not in routine_event or "n_points" not in routine_event):
                raise ValueError(
                    "An add-points event must contain 'layout' and 'n_points'."
                )

            layout = routine_event["layout"]

            if layout not in POINT_LAYOUTS:
                raise ValueError(
                    f"Unknown point layout {layout!r}: use one of "
                    f"{POINT_LAYOUTS}."
                )

            n_points = _require_integer(
                routine_event["n_points"],
                name=f"birth event at step {step} n_points",
                minimum=1,
            )

            positions = generate_point_positions(
                layout=layout,
                n_points=n_points,
                study_x_min=study_x_min,
                study_x_max=study_x_max,
                study_y_min=study_y_min,
                study_y_max=study_y_max,
                margin=point_margin,
                rng=event_rng,
            )

            for position in positions:
                point_idx = next_point_idx
                next_point_idx += 1
                priority = int(
                    event_rng.integers(
                        current_priority_min,
                        current_priority_max + 1,
                    )
                )

                point_events.append(
                    {
                        "step": step,
                        "action": "create",
                        "point_idx": point_idx,
                        "position": (
                            float(position[0]),
                            float(position[1]),
                        ),
                        "priority": priority,
                    }
                )
                active_point_idxs.append(point_idx)

        elif event_type == "remove_points":
            if "point_idxs" not in routine_event:
                raise ValueError(
                    "A remove-points event must contain 'point_idxs'."
                )

            point_idxs = routine_event["point_idxs"]

            if not isinstance(point_idxs, (list, tuple)):
                raise TypeError(
                    "Removed point identifiers must be a list or tuple."
                )

            validated_point_idxs = [
                _require_integer(
                    point_idx,
                    name=f"removed point identifier at step {step}",
                    minimum=0,
                )
                for point_idx in point_idxs
            ]

            if len(set(validated_point_idxs)) != len(validated_point_idxs):
                raise ValueError(
                    "A remove-points event cannot repeat an identifier."
                )

            inactive_point_idxs = (
                set(validated_point_idxs) - set(active_point_idxs)
            )

            if inactive_point_idxs:
                raise ValueError(
                    "A remove-points event references inactive point "
                    f"identifiers: {sorted(inactive_point_idxs)}."
                )

            for point_idx in validated_point_idxs:
                point_events.append(
                    {
                        "step": step,
                        "action": "remove",
                        "point_idx": point_idx,
                    }
                )

            removed_point_idxs = set(validated_point_idxs)
            active_point_idxs = [
                point_idx
                for point_idx in active_point_idxs
                if point_idx not in removed_point_idxs
            ]

        elif event_type == "change_priority_range":

            if ("min_priority" not in routine_event or "max_priority" not in routine_event):
                raise ValueError("A priority-range event must contain 'min_priority' and 'max_priority'.")

            new_min_priority = _require_integer(routine_event["min_priority"], name=f"priority minimum at step {step}", minimum=1)
            new_max_priority = _require_integer(routine_event["max_priority"], name=f"priority maximum at step {step}", minimum=1)

            if new_min_priority > new_max_priority:
                raise ValueError(f"Priority minimum at step {step} cannot be greater than its maximum.")

            # The new range also applies to points created by later reconfiguration events.
            current_priority_min = new_min_priority
            current_priority_max = new_max_priority

            for point_idx in active_point_idxs:
                new_priority = int(event_rng.integers(current_priority_min, current_priority_max + 1))

                point_events.append(
                    {
                        "step": step,
                        "action": "change_priority",
                        "point_idx": point_idx,
                        "new_priority": new_priority,
                    }
                )

        elif event_type == "set_priorities":
            if "priorities" not in routine_event:
                raise ValueError(
                    "An explicit-priority event must contain 'priorities'."
                )

            priorities = routine_event["priorities"]

            if not isinstance(priorities, (list, tuple)):
                raise TypeError(
                    "Explicit point priorities must be a list or tuple."
                )

            if len(priorities) != len(active_point_idxs):
                raise ValueError(
                    "The number of explicit priorities must match the number of active points."
                )

            validated_priorities = [
                _require_integer(
                    priority,
                    name=f"priority {position} at step {step}",
                    minimum=1,
                )
                for position, priority in enumerate(priorities)
            ]

            current_priority_min = min(validated_priorities)
            current_priority_max = max(validated_priorities)

            for point_idx, new_priority in zip(
                active_point_idxs,
                validated_priorities,
            ):
                point_events.append(
                    {
                        "step": step,
                        "action": "change_priority",
                        "point_idx": point_idx,
                        "new_priority": new_priority,
                    }
                )

    return point_events
