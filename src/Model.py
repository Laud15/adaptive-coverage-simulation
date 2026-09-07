import threading

import numpy as np
import pandas as pd
import mesa
from mesa.experimental.continuous_space import ContinuousSpace

from Agents import FixedWingDrone, QuadcopterDrone, TargetAgent
from PointsLayout import POINT_LAYOUTS, generate_point_positions
from PointScenario import build_point_events, get_point_routine

DRONE_CLASS_BY_TYPE = {
    "fixed_wing": FixedWingDrone,
    "quadcopter": QuadcopterDrone,
}

# --- THREAD-SAFE DATA COLLECTION CLASS ---
class ThreadSafeDataCollector(mesa.DataCollector):
    """Custom DataCollector version that prevents race conditions.

    When SolaraViz reads data to update the plots while the engine
    is writing the next step in the background, dataframe columns
    may temporarily have different lengths (e.g., 531 vs 530).
    This class aligns the columns on the fly by truncating the longest to the
    minimum common length, preventing a Pandas/Matplotlib crash.
    """

    def get_model_vars_dataframe(self):
        if not self.model_vars:
            return pd.DataFrame()

        # Find the minimum common length among all collected data
        min_len = min(len(values) for values in self.model_vars.values())

        # Temporarily truncate each column to that length
        safe_data = {key: values[:min_len] for key, values in self.model_vars.items()}
        return pd.DataFrame(safe_data)


class CoverageModel(mesa.Model):
    """The world: a closed rectangle containing points of interest and drones.

    Model responsibilities (in implementation order):
      1. create the space, initial points, and drones;
      2. apply scheduled point events and execute the synchronous drone phases;
      3. update point occupancy and collect metrics.
    """

    def __init__ (
        self,
        # --- ENVIRONMENT: territory configuration ---
        width=100.0, # territory width (x axis), in simulation units
        height=100.0, # territory height (y axis)
        n_drones=40, # number of drones, fixed throughout the simulation
        n_points=12, # number of points of interest created initially
        max_priority=3, # maximum randomly assigned quota: each point requests between 1 and 3 drones
        point_margin=0.0, # optional point-center margin inside the study area
        flight_buffer=None, # None -> coverage_radius + speed on every side of the study area
        point_layout="random",  # random | clusters | dispersed | circle | edges | central

        # --- DYNAMIC POINT EVENTS ---
        point_events=None,  # optional precompiled low-level point calendar
        point_routine="static",  # named high-level routine
        event_seed=0,  # seed used only to compile dynamic point events

        # --- DEPLOYMENT: drone starting locations ---
        deployment="dispersed", # dispersed | base | top | bottom | left | right
        deployment_noise=1.0, # dispersion around the starting base/side

        # --- DRONE TYPE ---
        drone_type="quadcopter", # "quadcopter" | "fixed_wing"

        # --- PHYSICAL SCALE (does not change dynamics, makes them interpretable) ---
        meters_per_unit=1.0,
        seconds_per_step=1.0,
 
        # --- DRONE GEOMETRY: distances, all in the same world units ---
        speed=1.0, # distance traveled at each step
        drone_sensing_radius=10.0, # radius within which it sees/communicates with other drones
        point_sensing_radius=10.0, # radius within which it perceives points
        separation=2.0, # below this distance another drone is "too close" and is avoided
        coverage_radius=8.0, # within this radius from a point, the drone is STATIONING
 
        # --- FORCE WEIGHTS: relative contribution of each force ---
        cohere=0.25, # strength of attraction toward the selected point
        separate=0.015, # strength of separation from drones that are too close
        match=0.05,# strength of alignment with the neighbors' average route
        boundary=1.2, # strength of the inward boundary force
        margin=12.0, # distance from the boundary at which the boundary force activates
        quadcopter_margin=2.0,# reduced margin: the quadcopter can turn in place

        # --- DECISION AND EXPLORATION ---
        beta=0.05, # travel cost: how much distance is penalized when selecting a point
        explore=0.2, # random steering strength when no point needing service is visible

        # --- FIXED-WING RELEASE DUE TO OVERCROWDING ---
        # Maximum random delay used by the fixed-wing target-release logic.
        # The quadcopter policy does not use this parameter.
        release_delay_max_steps=5,

        # --- QUADCOPTER COORDINATION ---
        # Small deviation applied by an explorer when it encounters a station that is already satisfied. 
        # It is expressed in degrees for immediate interpretation.
        avoid_angle_degrees=10.0,

        # How far inside the coverage boundary supports stop.
        support_inset=2.0,

        collect_agent_data=False, # When True, the datacollector also records per-agent data
        seed=None, # random seed: same seed = identical simulation
    ):
        # --- SEED ---
        # Mesa 3.5.1 expects rng=, NOT seed=: 'seed=' still works but emits a FutureWarning.
        # After this line, the following exist:
        #   self.rng -> NumPy Generator (used by drone exploration)
        #   self.random -> stdlib random.Random (required by ContinuousSpace)
        # both derived from the same seed: same seed = same simulation.
        super().__init__(rng=seed)

        # Solara may render the continuous space while a background step is creating or removing points.
        self.space_update_lock = threading.RLock()

        # --- VALIDITY CONSTRAINTS ---

        if drone_type not in DRONE_CLASS_BY_TYPE:
            raise ValueError(f"Unknown drone_type='{drone_type}': use {tuple(DRONE_CLASS_BY_TYPE)}.")

        if point_layout not in POINT_LAYOUTS:
            raise ValueError(
                f"Unknown point_layout='{point_layout}': "
                f"use {POINT_LAYOUTS}."
            )

        study_width = float(width)
        study_height = float(height)

        if study_width <= 0 or study_height <= 0:
            raise ValueError("width and height must be > 0.")


        if speed <= 0:
            raise ValueError("speed must be > 0.")

        if coverage_radius <= 0:
            raise ValueError("coverage_radius must be > 0.")

        # The study area contains the points of interest.
        # The surrounding flight space may also contain initial drone deployments and provides a buffer for stationing-zone exits.
        minimum_flight_buffer = float(coverage_radius) + float(speed)

        if flight_buffer is None:
            flight_buffer = minimum_flight_buffer
        else:
            flight_buffer = float(flight_buffer)
            if flight_buffer + 1e-9 < minimum_flight_buffer:
                raise ValueError(
                    f"flight_buffer ({flight_buffer}) < coverage_radius + speed "
                    f"({minimum_flight_buffer}): stationing zones need one full "
                    "exit step before the world boundary."
                )

        point_margin = float(point_margin)
        if point_margin < 0 or 2 * point_margin >= min(study_width, study_height):
            raise ValueError(
                f"Invalid point_margin ({point_margin}) for a "
                f"{study_width}x{study_height} study area."
            )

        flight_width = study_width + (2.0 * flight_buffer)
        flight_height = study_height + (2.0 * flight_buffer)

        # A drone covering a point must also perceive it.
        if coverage_radius > point_sensing_radius:
            raise ValueError(
                f"coverage_radius ({coverage_radius}) > "
                f"point_sensing_radius ({point_sensing_radius}): "
                "a drone could cover a point without perceiving it."
            )

        # A drone that perceives a staffed point must also perceive its owner at the center.
        if drone_type == "quadcopter" and drone_sensing_radius + 1e-9 < point_sensing_radius:
            raise ValueError(
                f"drone_sensing_radius ({drone_sensing_radius}) < "
                f"point_sensing_radius ({point_sensing_radius}): "
                "a drone could perceive a point without perceiving its owner."
            )

        # Every drone inside the separation distance must already be perceived.
        if drone_sensing_radius + 1e-9 < separation:
            raise ValueError(
                f"drone_sensing_radius ({drone_sensing_radius}) < "
                f"separation ({separation}): "
                "a drone could enter the separation range before being perceived."
            )

        # The turning-radius constraint applies ONLY to the fixed-wing drone.
        if drone_type == "fixed_wing":
            if cohere <= 0:
                raise ValueError("cohere must be > 0 for the fixed-wing drone.")
            self.turning_radius = speed / cohere
            if self.turning_radius >= coverage_radius:
                raise ValueError(
                    f"turning radius speed/cohere = {self.turning_radius:.2f} >= "
                    f"coverage_radius = {coverage_radius}: the fixed-wing drone would orbit outside the zone."
                )
        else:
            self.turning_radius = None

        for margin_name, margin_value in (("margin", margin),("quadcopter_margin", quadcopter_margin)):
            if margin_value <= 0 or 2 * margin_value >= min(width, height):
                raise ValueError(
                    f"Invalid {margin_name} ({margin_value}) for a "
                    f"{width}x{height} world: it must be positive and the boundary "
                    "force must not act everywhere."
                )

        if 2 * margin >= min(width, height):
            raise ValueError(
                f"margin ({margin}) is too large for a {width}x{height} world: "
                "the boundary force would act everywhere."
            )

        if meters_per_unit <= 0 or seconds_per_step <= 0:
            raise ValueError("meters_per_unit and seconds_per_step must be > 0.")

        if release_delay_max_steps < 0:
            raise ValueError("release_delay_max_steps must be >= 0.")

        if avoid_angle_degrees < 0:
            raise ValueError("avoid_angle_degrees must be >= 0.")

        if drone_type == "quadcopter" and not (0.0 < support_inset < coverage_radius):
            raise ValueError("support_inset must be > 0 and < coverage_radius.")

        # --- GEOMETRY ---
        # width/height must be ON THE MODEL because Drone._boundary_force() reads self.model.width / self.model.height,
        # and the final np.clip in Agents.py reads the same two names.
        # If they are named differently (self.available_width...), the drone fails with AttributeError at the first step,
        # not during construction: the error arrives late and appears unrelated to its cause.
        self.study_width = study_width
        self.study_height = study_height
        self.flight_buffer = float(flight_buffer)

        self.study_x_min = self.flight_buffer
        self.study_x_max = self.study_x_min + self.study_width
        self.study_y_min = self.flight_buffer
        self.study_y_max = self.study_y_min + self.study_height

        self.width = flight_width
        self.height = flight_height
        self.n_drones = int(n_drones)
        self.n_points = int(n_points)
        self.drone_type = drone_type
        self.drone_class = DRONE_CLASS_BY_TYPE[drone_type]
        self.point_layout = point_layout
        self.point_margin = float(point_margin)

        # Inter-drone perception and communication coincide by model assumption.
        self.communication_radius = float(drone_sensing_radius)

        # Physical scale: currently explicit metadata that does not alter the equations.
        self.meters_per_unit = float(meters_per_unit)
        self.seconds_per_step = float(seconds_per_step)
        self.simulated_time_s = 0.0
        self.real_speed_m_s = float(speed) * self.meters_per_unit / self.seconds_per_step

        # --- CONTINUOUS SPACE ---
        # dimensions: one row per axis -> [[x_min, x_max], [y_min, y_max]].
        # THE ORIGIN MUST BE 0: _boundary_force() compares position with 'margin' assuming the lower boundary is 0, and clipping uses [eps, width-eps].
        # With [[50, 150], ...], drones would behave as if the boundary were at 0 -> incorrect boundary force and ValueError from the space.
        # torus=False: closed, bounded territory (a square, not Pac-Man).
        # random=self.random: if omitted, Mesa emits UserWarning and uses an unseeded RNG -> non-reproducible simulation.
        # n_agents: only a pre-allocation hint for the internal position array. 
        # If incorrect, the array resizes automatically: it costs some time but is not a bug.
        self.space = ContinuousSpace(
            [[0.0, self.width], [0.0, self.height]],
            torus=False,
            random=self.random,
            n_agents=self.n_drones + self.n_points,
        )
        # Solara replaces renderer.space when it rebuilds the model.
        # Keeping the lock on the space ensures that the renderer always acquires the lock belonging to the currently displayed model.
        self.space.space_update_lock = self.space_update_lock

        # --- POINTS OF INTEREST: INITIAL CONDITION ---
        # Like drone "deployment", point_layout defines the initial world condition.
        # Later reconfiguration events may independently select another layout.
        # All modes use self.rng: same seed + same parameters = same territory, even when the geometry is pseudorandom.
        point_positions = self._generate_point_positions(layout=point_layout, margin=point_margin)

        # Priority = desired ABSOLUTE DRONE QUOTA, therefore expressed as INTEGER VALUES (the type remains float).
        # With a fractional quota (e.g., 2.5), the point stops attracting at the third drone,
        # thus CONSUMING 3 while declaring 2.5: total_demand and unavoidable_deficit below would be underestimated,
        # and comparison with the centralized oracle would measure a difference caused only by rounding.
        # With integer quotas, the difference is zero.
        # NOTE for integers(): the upper bound is EXCLUDED. integers(1, 3) returns 1 or 2, never 3 -> max_priority + 1 is required.
        point_priorities = np.zeros(self.n_points)
        for i in range(self.n_points):
            point_priorities[i] = self.rng.integers(1, max_priority + 1)

        # create_agents(model, n, *args, **kwargs): ALWAYS passes model first, 
        # which is why TargetAgent.__init__ begins with 'model' and then reverses the order in super().__init__(space, model).
        # An argument is DISTRIBUTED (one per agent) if it is a list/tuple/ndarray of exactly length n; otherwise, the same value is REPEATED for all agents.
        # Therefore self.space (an object, not a sequence) reaches all agents unchanged.
        # Therefore positions are an (n, 2) array, not a shared tuple:
        # with n=2, an (x, y) tuple would be mistaken for "one position per agent," and the two points would receive position=x and position=y.
        # list(...) because create_agents returns an AgentSet: 
        # freeze it into an ordered, stable list, which is what the metrics require.
        self.target_agents = list(
            TargetAgent.create_agents(
                self,
                self.n_points,
                self.space,
                position=point_positions,
                priority=point_priorities,
            )
        )

        # Stable point identifier used by dynamic events and optional per-agent data.
        # It is separate from Mesa's unique_id, which is shared by every agent type.
        self.target_agents_by_idx = {}

        for idx, point in enumerate(self.target_agents):
            point.idx = idx
            self.target_agents_by_idx[idx] = point

        # First identifier available for a point created during the simulation.
        # Identifiers of removed points will not be reused.
        self._next_target_idx = len(self.target_agents)

        # --- HIGH-LEVEL POINT ROUTINE COMPILATION ---

        # Resolve the selected high-level routine before constructing the low-level event calendar.
        routine = get_point_routine(point_routine)

        # An explicit low-level calendar and a non-static named routine would describe two competing event sources.
        if point_events is not None and point_routine != "static":
            raise ValueError("Use either point_events or a non-static point_routine not both.")

        if point_events is None:
            point_events = build_point_events(
                routine=routine,
                initial_n_points=self.n_points,
                initial_max_priority=max_priority,
                study_x_min=self.study_x_min,
                study_x_max=self.study_x_max,
                study_y_min=self.study_y_min,
                study_y_max=self.study_y_max,
                point_margin=self.point_margin,
                event_seed=event_seed,
            )

        # Retain the scenario metadata for visualization, diagnostics, and reproducible experiment records.
        self.point_routine = point_routine
        self.event_seed = event_seed

        # --- LOW-LEVEL POINT EVENT CALENDAR ---
        # Events are grouped by step for constant-time lookup during the environmental-update phase.
        self.point_events_by_step = {}

        if point_events is None:
            point_events = ()

        required_fields_by_action = {
            "create": {"position", "priority"},
            "change_priority": {"point_idx", "new_priority"},
            "remove": {"point_idx"},
        }

        for event in point_events:
            if not isinstance(event, dict):
                raise TypeError("Each point event must be represented by a dictionary.")
            
            if "step" not in event or "action" not in event:
                raise ValueError("Each point event must contain 'step' and 'action'.")

            step = event["step"]

            if isinstance(step, bool) or not isinstance(step,(int, np.integer)):
                raise TypeError("Point event step must be an integer.")

            step = int(step)

            if step < 1:
                raise ValueError("Point event step must be at least 1.")

            action = event["action"]

            if (not isinstance(action, str) or action not in required_fields_by_action):
                raise ValueError("Point event action must be 'create', 'change_priority', or 'remove'.")

            missing_fields = (required_fields_by_action[action] - event.keys())

            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(f"Point event at step {step} is missing: {missing}.")

            # Store an independent top-level copy so the model does not alter
            # the scenario dictionary supplied by the experiment.
            stored_event = dict(event)
            stored_event["step"] = step

            self.point_events_by_step.setdefault(step, []).append(stored_event)


        # --- DRONES: INITIAL POSITIONS AND DIRECTIONS ---
        # Side-based modes distribute drones along one side and orient them inward initially. 
        # The small noise is orthogonal to the boundary and prevents all drones from being placed at the exact same coordinate.
        drone_positions = np.zeros((self.n_drones, 2))
        drone_directions = np.zeros((self.n_drones, 2))

        if deployment == "dispersed":
            for i in range(self.n_drones):
                drone_positions[i, 0] = self.rng.uniform(0.0, self.width)
                drone_positions[i, 1] = self.rng.uniform(0.0, self.height)
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                drone_directions[i] = [np.cos(angle), np.sin(angle)]

        elif deployment == "base":
            base_x = self.width / 2.0
            base_y = self.height / 2.0
            for i in range(self.n_drones):
                drone_positions[i, 0] = base_x + self.rng.normal(0.0, deployment_noise)
                drone_positions[i, 1] = base_y + self.rng.normal(0.0, deployment_noise)
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                drone_directions[i] = [np.cos(angle), np.sin(angle)]

        elif deployment in ("top", "bottom", "left", "right"):
            for i in range(self.n_drones):
                offset = abs(self.rng.normal(0.0, deployment_noise))

                if deployment == "left":
                    drone_positions[i] = [min(offset, self.width * 0.05), self.rng.uniform(0.0, self.height)]
                    drone_directions[i] = [1.0, 0.0]
                elif deployment == "right":
                    drone_positions[i] = [self.width - min(offset, self.width * 0.05), self.rng.uniform(0.0, self.height)]
                    drone_directions[i] = [-1.0, 0.0]
                elif deployment == "bottom":
                    drone_positions[i] = [self.rng.uniform(0.0, self.width), min(offset, self.height * 0.05)]
                    drone_directions[i] = [0.0, 1.0]
                else:  # top
                    drone_positions[i] = [self.rng.uniform(0.0, self.width), self.height - min(offset, self.height * 0.05)]
                    drone_directions[i] = [0.0, -1.0]
        else:
            raise ValueError(
                f"Unknown deployment='{deployment}': use 'dispersed', 'base', "
                "'top', 'bottom', 'left', or 'right'."
            )

        # Safety net: all positions must be strictly internal.
        eps = 1e-6
        drone_positions[:, 0] = np.clip(drone_positions[:, 0], eps, self.width - eps)
        drone_positions[:, 1] = np.clip(drone_positions[:, 1], eps, self.height - eps)

        # Positions and directions are EXPLICIT (n, 2) arrays, never shared tuples:
        # with n_drones=2, a tuple would be mistaken for "one value per agent."
        # The same _boundary_force() is used by both platforms, 
        # but with different margins: 
        #   -the fixed-wing drone must anticipate the turn, 
        #   -the quadcopter can react in the final steps near the boundary.
        boundary_margin = (quadcopter_margin if self.drone_class is QuadcopterDrone else margin)

        # Shared parameters are passed to both subclasses;
        # the parameter for deviation from satisfied stations is added only to the quadcopter.
        drone_parameters = dict(
            position=drone_positions,
            direction=drone_directions,
            speed=speed,
            drone_sensing_radius=drone_sensing_radius,
            point_sensing_radius=point_sensing_radius,
            separation=separation,
            coverage_radius=coverage_radius,
            cohere=cohere,
            separate=separate,
            match=match,
            boundary=boundary,
            margin=boundary_margin,
            beta=beta,
            explore=explore,
            release_delay_max_steps=release_delay_max_steps,
        )
        if self.drone_class is QuadcopterDrone:
            drone_parameters.update(avoid_angle_degrees=avoid_angle_degrees, support_inset=support_inset)

        self.drone_agents = list(
            self.drone_class.create_agents(
                self,
                self.n_drones,
                self.space,
                **drone_parameters,
            )
        )

        # The model keeps a copy of the parameters it needs independently:
        # coverage_radius is the radius used to count occupancy in block 4.
        self.coverage_radius = float(coverage_radius)
        self.drone_sensing_radius = float(drone_sensing_radius)
        self.point_sensing_radius = float(point_sensing_radius)
        self.boundary_margin = float(boundary_margin)

        # --- STRUCTURAL DIAGNOSTICS ---
        # These quantities depend on the currently active points and must be recomputed after: births, deaths, and priority changes.
        self._refresh_point_diagnostics()

        # Initial snapshot. For fixed-wing drones, coverage is geometric; 
        # for the quadcopter, a drone counts only after the OWNER/SUPPORT election, 
        # it is correct not to count a quadcopter spawned in the zone while it still has no stationary role.
        self.update_occupancy()

        # --- DATA COLLECTION ---
        # Reporters are the METHODS below, passed as CoverageModel.name (the function, not its result: no parentheses).
        # Mesa invokes them at each collect, passing the model. Regular methods rather than lambdas for two reasons:
        #  1) they can be called manually from a test script,
        #  2) their name appears in a traceback instead of an anonymous "<lambda>."
        # "total_demand" and "unavoidable_deficit" use the STRING form:
        # Mesa reads the corresponding model attributes. 
        # Both values are recomputed whenever the active points or their quotas change.
        model_reporters = {
            "residual_deficit": CoverageModel.residual_deficit,
            "normalized_deficit": CoverageModel.normalized_deficit,
            "satisfied_points": CoverageModel.satisfied_points,
            "overservice": CoverageModel.overservice,
            "idle_drones": CoverageModel.idle_drones,
            "exploring_drones": CoverageModel.exploring_drones,
            "active_points": CoverageModel.active_points, 
            "total_demand": "total_demand",
            "unavoidable_deficit": "unavoidable_deficit",
            "simulated_time_s": "simulated_time_s",
        }

        # Per-agent data is disabled by default:
        # it produces n_drones + M(t) rows at every step (52 x 600 = 31,200 rows in a static 40-drone, 12-point run),
        # and explodes in a sweep with dozens of combinations. Enable it when inspecting one simulation, not when running hundreds.
        agent_type_reporters = None
        if collect_agent_data:
            drone_reporters = {
                "n_covered": "n_covered",
                "exploring": "exploring",
                "drone_type": "drone_type",
                "moving": "moving",
                "release_wait_remaining": "release_wait_remaining",
            }
            if self.drone_class is QuadcopterDrone:
                drone_reporters["station_role"] = "station_role"

            agent_type_reporters = {
                TargetAgent: {"idx": "idx", "priority": "priority", "occupancy": "occupancy"},
                self.drone_class: drone_reporters,
            }

        self.datacollector = ThreadSafeDataCollector(model_reporters=model_reporters, agenttype_reporters=agent_type_reporters)

        # First row: state at t=0, before anything moves.
        # It provides a transient reference: without it, the first value is already the result of one step, with no initial baseline.
        self.datacollector.collect(self)

    # ------------------------------------------------------------------
    # GENERATION OF INITIAL POINT CONDITIONS
    # ------------------------------------------------------------------

    def _generate_point_positions(self, layout, margin):
        """Generate the positions of the initial points of interest.

        This wrapper supplies the current model geometry and random-number
        generator to the shared point-layout function.
        """
        return generate_point_positions(
            layout=layout,
            n_points=self.n_points,
            study_x_min=self.study_x_min,
            study_x_max=self.study_x_max,
            study_y_min=self.study_y_min,
            study_y_max=self.study_y_max,
            margin=margin,
            rng=self.rng,
        )

    # ------------------------------------------------------------------
    # DYNAMIC POINT MANAGEMENT
    # ------------------------------------------------------------------

    def _apply_point_events(self):
        """Apply the point events scheduled for the current step."""

        # Direct dictionary lookup: if no event is scheduled for this step,
        # use an empty tuple and perform no operation.
        events = self.point_events_by_step.get(self.steps, ())

        # Events scheduled at the same step are applied in the order in which they were provided in point_events.
        for event in events:
            action = event["action"]

            if action == "create":

                self.create_point(
                    position=event["position"],
                    priority=event["priority"],
                    point_idx=event.get("point_idx"),
                )

            elif action == "change_priority":
                self.change_point_priority(point_idx=event["point_idx"], new_priority=event["new_priority"])

            elif action == "remove":
                self.remove_point(point_idx=event["point_idx"])

            else:
                # This should be unreachable because actions are validated
                # when the event calendar is constructed.
                raise RuntimeError(f"Unsupported point event action: {action!r}.")

            
    def create_point(self, position, priority, point_idx=None):
        """Create and register a new active point of interest.
        
        The point receives a stable point-specific identifier.
        """
        # A point quota must remain a positive integer. 
        # It is stored as a float for compatibility with the existing TargetAgent implementation.
        try:
            priority_value = float(priority)
        except (TypeError, ValueError):
            raise ValueError("Point priority must be a positive integer.") from None

        if (not np.isfinite(priority_value) or priority_value <= 0 or not priority_value.is_integer()):
            raise ValueError("Point priority must be a positive integer.")


        # Convert and validate the position before constructing the agent,
        # because construction automatically registers it with Mesa.
        try:
            position_array = np.asarray(position, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("Point position must contain two finite coordinates.") from None

        if position_array.shape != (2,) or not np.all(np.isfinite(position_array)):
            raise ValueError("Point position must contain two finite coordinates.")

        # Dynamic points follow the same center-generation bounds as initial points,
        # including point_margin.
        x_min = self.study_x_min + self.point_margin
        x_max = self.study_x_max - self.point_margin
        y_min = self.study_y_min + self.point_margin
        y_max = self.study_y_max - self.point_margin

        if not(x_min <= position_array[0] <= x_max and y_min <= position_array[1] <= y_max):
            raise ValueError("Point position must lie inside the allowed study area.")

        # If the scenario does not provide an identifier, use the next one.
        if point_idx is None:
            point_idx = self._next_target_idx
        else:
            if isinstance(point_idx, bool) or not isinstance(point_idx, (int, np.integer)):
                raise TypeError("point_idx must be an integer.")

            point_idx = int(point_idx)

            if point_idx < 0:
                raise ValueError("point_idx must be non-negative.")

            if point_idx in self.target_agents_by_idx:
                raise ValueError(f"A point with idx={point_idx} is already active.")

            # Point identifiers are monotonically increasing. 
            # An identifier lower than the next available one may have been used previously.
            if point_idx < self._next_target_idx:
                raise ValueError(f"Point idx={point_idx} cannot be reused.")

        # TargetAgent construction automatically registers the new agent in both the Mesa model and the continuous space.
        point = TargetAgent(model=self, space=self.space, position=position_array, priority=priority_value)

        # Register the same object in the project-specific collections.
        point.idx = point_idx
        self.target_agents.append(point)
        self.target_agents_by_idx[point_idx] = point

        # Keep the automatic counter ahead of every explicitly assigned ID.
        self._next_target_idx = max(self._next_target_idx, point_idx + 1)

        self._refresh_point_diagnostics()
        return point


    def change_point_priority(self, point_idx, new_priority):
        """Change the requested quota of an active point.

        The point keeps the same identity and position. Only its requested
        quota and the model-level diagnostics are updated.
        """
        # A boolean must not be silently interpreted as point 0 or point 1.
        if isinstance(point_idx, bool) or not isinstance(point_idx,(int, np.integer)):
            raise TypeError("point_idx must be an integer.")

        point_idx = int(point_idx)

        # Only currently active points can change their quota.
        if point_idx not in self.target_agents_by_idx:
            raise KeyError(f"No active point with idx={point_idx}.")

        # The requested quota must remain a positive integer.
        try:
            priority_value = float(new_priority)
        except (TypeError, ValueError):
            raise ValueError("Point priority must be a positive integer.") from None

        if (not np.isfinite(priority_value) or priority_value <= 0 or not priority_value.is_integer()):
            raise ValueError("Point priority must be a positive integer.")

        point = self.target_agents_by_idx[point_idx]
        point.priority = priority_value

        # Total demand and its derived diagnostics change with the quota.
        self._refresh_point_diagnostics()


    def remove_point(self, point_idx):
        """Remove an active point and clear every direct drone association.

        This operation must run during the environmental-event phase, before
        perception starts for the current step.
        """
        # A boolean must not be silently interpreted as point 0 or point 1.
        if isinstance(point_idx, bool) or not isinstance(point_idx,(int, np.integer)):
            raise TypeError("point_idx must be an integer.")

        point_idx = int(point_idx)

        # Only an active point can be removed.
        if point_idx not in self.target_agents_by_idx:
            raise KeyError(f"No active point with idx={point_idx}.")

        point = self.target_agents_by_idx[point_idx]

        # This should never fail: the list and dictionary are maintained together. 
        # Check before modifying anything to avoid partial removal.
        if point not in self.target_agents:
            raise RuntimeError("Point registry and active-point list are inconsistent.")

        # Clear current and buffered references before removing the point from Mesa. 
        # No drone may access point.position after point.remove().
        for drone in self.drone_agents:
            drone.handle_removed_point(point)

        # Remove the point from the project-specific active collections.
        self.target_agents.remove(point)
        del self.target_agents_by_idx[point_idx]

        # ContinuousSpaceAgent.remove() deregisters the agent from both the
        # Mesa model and the continuous space.
        point.remove()

        # Demand and geometric diagnostics now refer only to active points.
        self._refresh_point_diagnostics()



    def _refresh_point_diagnostics(self):
        """Recompute diagnostics derived from the currently active points.

        These values are global ground truth used for metrics and analysis.
        They are not available to the decentralized drone policy.
        """
        # Total requested quota of all currently active points.
        self.total_demand = 0.0

        for point in self.target_agents:
            self.total_demand += point.priority

        # Lower bound under the assumption that each drone can contribute to at most one point. 
        # It is not guaranteed when coverage zones overlap, because one drone can contribute to multiple points.
        self.unavoidable_deficit = max(0.0, self.total_demand - self.n_drones)

        # Count pairs of points whose coverage zones overlap.
        self.overlapping_zones = 0
        n_active_points = len(self.target_agents)

        for i in range(n_active_points):
            for j in range(i + 1, n_active_points):
                distance = np.linalg.norm(self.target_agents[i].position - self.target_agents[j].position)

                if distance < 2.0 * self.coverage_radius:
                    self.overlapping_zones += 1

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------

    def active_points(self):
        """Return the number of points currently active in the model."""
        return len(self.target_agents)

    def residual_deficit(self):
        """Total number of drones missing for every point to reach its quota.

        This is THE metric: the one coordination must minimize.
        The max(0, ...) is not cosmetic. Without it, an over-served point would make a
        NEGATIVE contribution that offsets an uncovered point elsewhere: a world with
        half the points empty and half crowded would appear perfect.
        """
        total = 0.0
        for point in self.target_agents:
            missing = point.priority - point.occupancy
            if missing > 0:
                total += missing
        return total

    def normalized_deficit(self):
        """Residual deficit as a fraction of total demand.

        Used to COMPARE runs with different territories: a deficit of 6 out of a demand
        of 26 and one of 6 out of a demand of 60 are not the same performance. In
        sweeps where n_points or max_priority vary, the raw deficit is not comparable
        across combinations: this one is.
        """
        if self.total_demand <= 0:
            return 0.0
        return self.residual_deficit() / self.total_demand

    def satisfied_points(self):
        """Number of points that have reached (or exceeded) their quota.

        Looks at the same quantity as the deficit, but by POINT COUNT rather than amount:
        it shows whether the system serves a few points well or many points halfway.
        Two configurations with the same residual deficit can differ greatly here.
        """
        n = 0
        for point in self.target_agents:
            if point.occupancy >= point.priority:
                n += 1
        return n

    def overservice(self):
        """Excess drones on points that are already fully served.

        This is the other side of waste, and is NOT the complement of idle drones: an
        idle drone stations at no point, while an over-serving drone stations at one
        that was already satisfied. Different waste, different remedies.
        """
        total = 0.0
        for point in self.target_agents:
            excess = point.occupancy - point.priority
            if excess > 0:
                total += excess
        return total

    def idle_drones(self):
        """Drones that are not currently stationed at any point."""
        n = 0
        for drone in self.drone_agents:
            if drone.n_covered == 0:
                n += 1
        return n

    def exploring_drones(self):
        """Drones with no point in need anywhere in view.

        This is a SUBSET of idle drones: an idle drone may be traveling towards a point
        it has already selected, while an exploring drone is not. The gap between the
        two figures is diagnostic: if they are almost equal, the problem is that drones
        do not FIND points (random exploration is weak); if they differ greatly, drones
        find the points but take too long to reach them.
        """
        n = 0
        for drone in self.drone_agents:
            if drone.exploring:
                n += 1
        return n

    def update_occupancy(self):
        """
        Recounts how many drones are currently stationed at each point.

        Points do not compute this themselves: a point does not know what surrounds it.
        The model, which sees everyone, does it and WRITES the result into each point.
        Drones do NOT read it to make decisions: they use their own local estimate built
        through perception and communication. ``occupancy`` remains the ground truth for
        metrics and visualization. For quadcopters, geometric presence alone is not
        enough: only OWNER and SUPPORT drones count, and they are stationary by policy.
        Fixed-wing drones instead retain the original geometric definition.
        """
        # First reset all drone counters: n_covered is rebuilt from zero at every step, not incremented indefinitely.
        for drone in self.drone_agents:
            drone.n_covered = 0 # n_covered indicates how many points that drone is physically covering.

        for point in self.target_agents:
            # One call per point: distances from THIS point to ALL drones.
            # The array is returned in the same order as self.drone_agents (verified by running it), so distances[i] is the distance of drone i.
            distances, _ = self.space.calculate_distances(point.position, agents=self.drone_agents)

            covering_drones = 0
            for i in range(len(self.drone_agents)):
                drone = self.drone_agents[i]

                # If it is outside coverage, it is not counted.
                if distances[i] > self.coverage_radius:
                    continue
                # For quadcopters, the role check is also applied.
                # A quadcopter is counted only if it is within coverage_radius AND station_role is owner or support.
                if (self.drone_class is QuadcopterDrone and getattr(drone, "station_role", None) not in ("owner", "support")):
                    continue

                covering_drones += 1
                # NOTE: a drone within the coverage_radius of TWO nearby points counts for both.
                # Therefore, the sum of occupancy values can exceed n_drones: this is intended (a drone between two zones really stations at both),
                # but it must be remembered when reading the metrics.
                drone.n_covered += 1 # how many points are covered by this drone

            point.occupancy = covering_drones # how many drones cover this point

    def step(self):
        """One world step, explicitly divided into phases.

        There is no real parallelism: each phase ends for ALL drones before the next one
        begins. ``shuffle_do`` only randomizes the internal order of the phase; methods
        are designed to write their own state, not the state of others.
        """
        drones = self.agents_by_type[self.drone_class]

        # 1. Apply exogenous point events before drones observe the environment.
        # Structural space updates must not overlap with Solara rendering.
        with self.space_update_lock:
            self._apply_point_events()

        # 2. Active points execute their per-step behavior.
        self.agents_by_type[TargetAgent].do("step")

        # 3. Everyone builds a local snapshot of the same spatial state.
        drones.shuffle_do("perceive")

        # 4. Everyone reads neighboring snapshots and builds their own estimate.
        drones.shuffle_do("communicate")

        # 5. Everyone chooses a target without yet modifying the current target/role.
        drones.shuffle_do("decide_target")

        # 6. This phase exists for all platforms: BaseDrone defines it as a no-op,
        #    while QuadcopterDrone specializes it with owner/support roles and release from overcrowding.
        drones.shuffle_do("decide_station")

        # 7. Decisions become current state.
        drones.shuffle_do("commit_decision")

        # 8. Physical movement.
        drones.shuffle_do("move")

        # 9. Ground truth: the model recounts actual stationing AFTER movement.
        self.update_occupancy()

        # 10. Physical time and measurement. The t=0 row was collected in __init__.
        self.simulated_time_s += self.seconds_per_step
        self.datacollector.collect(self)
