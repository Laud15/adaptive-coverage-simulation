import numpy as np
import pandas as pd
import mesa
from mesa.experimental.continuous_space import ContinuousSpace

from Agents import FixedWingDrone, QuadcopterDrone, TargetAgent


DRONE_CLASS_BY_TYPE = {
    "fixed_wing": FixedWingDrone,
    "quadcopter": QuadcopterDrone,
}

# Available initial conditions for points of interest.
# These are geometric scenarios, not adaptive policies: they are selected once in __init__.
POINT_LAYOUTS = (
    "random",
    "clusters",
    "dispersed",
    "circle",
    "edges",
    "central",
)

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
      1. create the space, points, and drones         
      2. update point occupancy at each step 
      3. collect deficit metrics       
    """

    def __init__ (
        self,
        # --- ENVIRONMENT: territory configuration ---
        width=100.0, # territory width (x axis), in simulation units
        height=100.0, # territory height (y axis)
        n_drones=40, # number of drones, fixed throughout the simulation
        n_points=12, # number of points of interest created initially
        max_priority=3, # maximum randomly assigned quota: each point requests between 1 and 3 drones
        point_margin=0.0, # controls how far point-of-interest centers are kept from boundaries when generated.
        point_layout="random",  # random | clusters | dispersed | circle | edges | central
 
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
        boundary=0.3, # strength of the inward boundary force
        margin=12.0, # distance from the boundary at which the boundary force activates
        quadcopter_margin=2.0,# reduced margin: the quadcopter can turn in place

        # --- DECISION AND EXPLORATION ---
        beta=0.05, # travel cost: how much distance is penalized when selecting a point
        explore=0.2, # random steering strength when no point needing service is visible

        # --- RELEASE DUE TO OVERCROWDING ---
        # BaseDrone/fixed wing: maximum random delay in the shared release logic.
        # Quadcopter: maximum amplitude of the pseudorandom component of the support timer.
        release_delay_max_steps=5,

        # --- QUADCOPTER COORDINATION ---
        # Small deviation applied by an explorer when it encounters a station
        # that is already satisfied. It is expressed in degrees for immediate interpretation.
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

        # --- VALIDITY CONSTRAINTS ---

        if drone_type not in DRONE_CLASS_BY_TYPE:
            raise ValueError(f"Unknown drone_type='{drone_type}': use {tuple(DRONE_CLASS_BY_TYPE)}.")

        if point_layout not in POINT_LAYOUTS:
            raise ValueError(
                f"Unknown point_layout='{point_layout}': "
                f"use {POINT_LAYOUTS}."
            )

        if point_margin < 0 or 2 * point_margin >= min(width, height):
            raise ValueError(
                f"Invalid point_margin ({point_margin}) for a {width}x{height} world."
            )

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
        self.width = float(width)
        self.height = float(height)
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

        # --- POINTS OF INTEREST: INITIAL CONDITION ---
        # Like drone "deployment", point layout is an initial world choice.
        # It does not change during the simulation.
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

        # idx = progressive index for colors and metrics. Assign it here rather than in the
        # constructor because a point does not know the order in which it was created.
        for i in range(self.n_points):
            self.target_agents[i].idx = i

        # --- STRUCTURAL DIAGNOSTICS ---
        # total_demand = sum of quotas = number of "drone slots" requested by the territory.
        # With priority as an absolute quota, if total_demand > n_drones, the system has a STRUCTURAL DEFICIT: no algorithm, not even the oracle.
        # Centralized, can reduce the residual deficit to zero.
        # unavoidable_deficit is the FLOOR of the main metric-
        self.total_demand = 0.0
        for point in self.target_agents:
            self.total_demand += point.priority
        self.unavoidable_deficit = max(0.0, self.total_demand - self.n_drones)

        # NOTE: unavoidable_deficit is a CONDITIONAL floor. 
        # It assumes that each drone provides at most one occupancy unit.
        # with overlapping zones, 10 drones may produce an occupancy of 13, and the residual deficit falls BELOW the floor. 
        # This is not a calculation error: the assumption no longer holds. 
        self.overlapping_zones = 0
        for i in range(self.n_points):
            for j in range(i + 1, self.n_points):
                d = np.linalg.norm(self.target_agents[i].position - self.target_agents[j].position)
                if d < 2.0 * coverage_radius:
                    self.overlapping_zones += 1

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

        # Initial snapshot. For fixed-wing drones, coverage is geometric; 
        # for the quadcopter, a drone counts only after the OWNER/SUPPORT election, 
        # it is correct not to count a quadcopter spawned in the zone while still FREE.
        self.update_occupancy()

        # --- DATA COLLECTION ---
        # Reporters are the METHODS below, passed as CoverageModel.name (the function, not its result: no parentheses).
        # Mesa invokes them at each collect, passing the model. Regular methods rather than lambdas for two reasons:
        #  1) they can be called manually from a test script
        #  2) their name appears in a traceback instead of an anonymous "<lambda>."
        # "unavoidable_deficit" instead uses the STRING form: Mesa reads the model attribute with the same name.
        #  It is a constant, but repeating it in every row allows the floor line to be plotted without retrieving it separately.
        model_reporters = {
            "residual_deficit": CoverageModel.residual_deficit,
            "normalized_deficit": CoverageModel.normalized_deficit,
            "satisfied_points": CoverageModel.satisfied_points,
            "overservice": CoverageModel.overservice,
            "idle_drones": CoverageModel.idle_drones,
            "exploring_drones": CoverageModel.exploring_drones,
            "unavoidable_deficit": "unavoidable_deficit",
            "simulated_time_s": "simulated_time_s",
        }

        # Per-agent data is disabled by default: 
        # it produces n_drones + n_points rows AT EVERY STEP (52 x 600 = 31,200 rows for a single run),
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
        """Builds the initial positions of the points of interest.

        The layouts are deliberately simple and readable: they create worlds with
        different geometries, rather than model a dynamic point-formation process.
        All randomness goes through ``self.rng``, so it is reproducible through
        ``seed``.
        """
        n = self.n_points
        positions = np.zeros((n, 2), dtype=float)
        if n == 0:
            return positions

        # Rectangle actually available after point_margin.
        x_min = float(margin)
        x_max = self.width - float(margin)
        y_min = float(margin)
        y_max = self.height - float(margin)
        width = x_max - x_min
        height = y_max - y_min

        if layout == "random":
            # Original baseline: independent points uniformly distributed across the territory.
            for i in range(n):
                positions[i, 0] = self.rng.uniform(x_min, x_max)
                positions[i, 1] = self.rng.uniform(y_min, y_max)

        elif layout == "clusters":
            # At most three clusters. Centers are random, but each cluster receives at least one point when n allows it.
            # Dispersion is 5% of the smallest dimension of the available rectangle.
            n_clusters = min(3, n)
            centers = np.zeros((n_clusters, 2), dtype=float)

            # Keep cluster centers away from the margin, so Gaussian noise is not clipped almost entirely on one side.
            padding_x = 0.15 * width
            padding_y = 0.15 * height
            for g in range(n_clusters):
                centers[g, 0] = self.rng.uniform(x_min + padding_x, x_max - padding_x)
                centers[g, 1] = self.rng.uniform(y_min + padding_y, y_max - padding_y)

            assignments = np.arange(n) % n_clusters
            self.rng.shuffle(assignments)
            sigma = 0.05 * min(width, height)

            for i in range(n):
                center = centers[assignments[i]]
                positions[i] = center + self.rng.normal(0.0, sigma, size=2)

        elif layout == "dispersed":
            # Divide the territory into cells and use only one position per cell.
            # A small jitter avoids a perfectly artificial grid, while keeping points much more separated than in the random baseline.
            aspect_ratio = width / height
            n_columns = max(1, int(np.ceil(np.sqrt(n * aspect_ratio))))
            n_rows = max(1, int(np.ceil(n / n_columns)))

            x_step = width / n_columns
            y_step = height / n_rows
            cells = []
            for row in range(n_rows):
                for column in range(n_columns):
                    cells.append(
                        [
                            x_min + (column + 0.5) * x_step,
                            y_min + (row + 0.5) * y_step,
                        ]
                    )

            cells = np.asarray(cells, dtype=float)
            self.rng.shuffle(cells)
            jitter_x = 0.15 * x_step
            jitter_y = 0.15 * y_step

            for i in range(n):
                positions[i, 0] = cells[i, 0] + self.rng.uniform(-jitter_x, jitter_x)
                positions[i, 1] = cells[i, 1] + self.rng.uniform(-jitter_y, jitter_y)

        elif layout == "circle":
            # Equally spaced points on a circle centered in the territory.
            # The initial angle is random: the shape remains a circle, but the seed determines the configuration's overall rotation.
            center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])
            radius = 0.35 * min(width, height)
            phase = self.rng.uniform(0.0, 2.0 * np.pi)

            for i in range(n):
                angle = phase + (2.0 * np.pi * i / n)
                positions[i] = center + radius * np.array(
                    [np.cos(angle), np.sin(angle)]
                )

        elif layout == "edges":
            # Distribution near the four boundaries. 
            # Side assignments are balanced and then shuffled, so they do not depend on the point index.
            band = 0.08 * min(width, height)
            sides = np.arange(n) % 4
            self.rng.shuffle(sides)

            for i, side in enumerate(sides):
                offset = self.rng.uniform(0.0, band)
                if side == 0:      # left
                    positions[i] = [x_min + offset, self.rng.uniform(y_min, y_max)]
                elif side == 1:    # right
                    positions[i] = [x_max - offset, self.rng.uniform(y_min, y_max)]
                elif side == 2:    # bottom
                    positions[i] = [self.rng.uniform(x_min, x_max), y_min + offset]
                else:              # top
                    positions[i] = [self.rng.uniform(x_min, x_max), y_max - offset]

        elif layout == "central":
            # All points fall inside the central rectangle, whose width/height is 30% of the available territory.
            # This is a central concentration, not a point-like cluster: points still retain some dispersion.
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            half_width = 0.15 * width
            half_height = 0.15 * height

            for i in range(n):
                positions[i, 0] = self.rng.uniform(center_x - half_width, center_x + half_width)
                positions[i, 1] = self.rng.uniform(center_y - half_height, center_y + half_height)

        else:
            # In practice this branch is protected by the guardrail in __init__,
            # but keeping it makes the function self-contained and easier to test directly.
            raise ValueError(f"Unrecognized point layout: {layout}")

        # Safety common to all layouts:
        # cluster noise/jitter cannot move points outside the rectangle allowed by point_margin.
        positions[:, 0] = np.clip(positions[:, 0], x_min, x_max)
        positions[:, 1] = np.clip(positions[:, 1], y_min, y_max)
        return positions

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------

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

        # 1. The territory changes.
        self.agents_by_type[TargetAgent].do("step")

        # 2. Everyone builds a local snapshot of the same spatial state.
        drones.shuffle_do("perceive")

        # 3. Everyone reads neighboring snapshots and builds their own estimate.
        drones.shuffle_do("communicate")

        # 4. Everyone chooses a target without yet modifying the current target/role.
        drones.shuffle_do("decide_target")

        # 5. This phase exists for all platforms: BaseDrone defines it as a no-op,
        #    while QuadcopterDrone specializes it with owner/support roles and release from overcrowding.
        drones.shuffle_do("decide_station")

        # 6. Decisions become current state.
        drones.shuffle_do("commit_decision")

        # 7. Physical movement.
        drones.shuffle_do("move")

        # 8. Ground truth: the model recounts actual stationing AFTER movement.
        self.update_occupancy()

        # 9. Physical time and measurement. The t=0 row was collected in __init__.
        self.simulated_time_s += self.seconds_per_step
        self.datacollector.collect(self)
