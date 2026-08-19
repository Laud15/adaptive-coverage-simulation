import numpy as np
from mesa.experimental.continuous_space import ContinuousSpaceAgent

EPS = 1e-9

class TargetAgent(ContinuousSpaceAgent):
    """Passive point of interest.

    ``occupancy`` is the ground truth calculated by the model. Drones do not read it
    to make decisions: it is used for metrics and visualization.
    """

    def __init__(self, model, space, position, priority=1.0):
        super().__init__(space=space, model=model)
        self.position = np.array(position, dtype=float)
        self.priority = float(priority)
        self.occupancy = 0

    def step(self):
        pass


class BaseDrone(ContinuousSpaceAgent):
    """Behavior shared by all UAV platforms in the model.

    This class contains only the parts shared by fixed-wing drones and quadcopters:
      * perception of points and drones;
      * local communication within ``drone_sensing_radius``;
      * local occupancy estimation through communication;
      * target selection;
      * separation, alignment, boundary avoidance, and exploration.

    Kinematics and stationing policies belong to the subclasses.
    """

    drone_type = "base"

    def __init__(
        self,
        model,
        space,
        position=(0, 0),
        direction=(1, 1),
        speed=1.0,
        drone_sensing_radius=10.0, # radius within which the drone sees and communicates with other drones
        point_sensing_radius=25.0, # point sensing radius
        separation=2.0, # Separation activates below this distance
        coverage_radius=8.0, # radius within which the drone starts covering a point
        cohere=0.25, # attraction toward the destination
        separate=0.015, # repulsion from drones that are too close
        match=0.05, # alignment with neighbors' directions
        boundary=0.3, # repulsion from boundaries
        margin=20.0, # Distance from the boundary at which _boundary_force() starts acting
        beta=0.05, # penalizes distance when selecting the drone's destination, utility = deficit - beta * distance
        explore=0.2, # Controls random direction variability during exploration
        release_delay_max_steps=5, # Amplitude of the random component of the wait before leaving due to overcrowding
    ):
        super().__init__(space=space, model=model)

        # --- initial geometric state ---
        self.position = np.array(position, dtype=float)
        self.direction = np.array(direction, dtype=float)
        self.direction = self._normalize(self.direction, fallback=[1.0, 0.0])

        # --- shared parameters ---
        self.speed = float(speed)
        self.drone_sensing_radius = float(drone_sensing_radius)  # = communication_radius by model assumption
        self.point_sensing_radius = float(point_sensing_radius)
        self.separation = float(separation)
        self.coverage_radius = float(coverage_radius)
        self.cohere_factor = float(cohere)
        self.separate_factor = float(separate)
        self.match_factor = float(match)
        self.boundary_factor = float(boundary)
        self.margin = float(margin)
        self.beta = float(beta)
        self.explore_factor = float(explore)
        self.release_delay_max_steps = int(release_delay_max_steps)

        # --- observable current state ---
        self.target = None
        self.n_covered = 0  # ground truth, written by the model
        self.exploring = False
        self.angle = float(np.degrees(np.arctan2(self.direction[1], self.direction[0])))
        self.moving = True # used for graphical representation, determines whether to draw the arrow showing the drone's direction

        # --- snapshot produced by the PERCEIVE phase ---
        self.neighbors = []
        self.neighbor_distances = []
        self.perceived_points = []
        self.perceived_point_distances = []

        # Local occupancy estimate, aligned with ``perceived_points``.
        # No point identifier enters the drone's knowledge:
        # perceived_point_occupancies[k] simply refers to perceived_points[k]
        # in the local snapshot of the current step.
        self.perceived_point_occupancies = []

        # --- current/next state decision ---
        self.planned_target = None
        self.planned_exploring = False

        # --- random wait before leaving an overcrowded point ---
        # None = no wait is active. When it starts, the value is drawn
        # only once and decremented at each step while overcrowding persists.
        self.release_wait_remaining = None

    # ------------------------------------------------------------------
    # BASE GEOMETRY
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(vector, fallback=None):
        vector = np.array(vector, dtype=float)
        magnitude = np.linalg.norm(vector)
        if magnitude > EPS:
            return vector / magnitude
        if fallback is not None:
            return np.array(fallback, dtype=float)
        return np.zeros(2)

    def _clip_position(self, position):
        """Prevents the drone position from leaving the simulation area."""
        eps = 1e-6
        lower_bound = np.array([eps, eps])
        upper_bound = np.array([self.model.width - eps, self.model.height - eps])
        return np.clip(position, lower_bound, upper_bound)

    def _update_angle(self):
        self.angle = float(np.degrees(np.arctan2(self.direction[1], self.direction[0])))

    def _boundary_force(self):
        """Force that keeps the drone within the boundaries."""
        force = np.zeros(2)
        for index, max_dimension in enumerate((self.model.width, self.model.height)):
            current_position = self.position[index]
            if current_position < self.margin:
                force[index] = (self.margin - current_position) / self.margin
            elif current_position > (max_dimension - self.margin):
                force[index] = (max_dimension - self.margin - current_position) / self.margin
        return force * self.boundary_factor

    # ------------------------------------------------------------------
    # PHASE 1 - PERCEPTION
    # ------------------------------------------------------------------

    def perceive(self):
        """Builds a local snapshot without modifying other agents."""
        # The two radii have independent semantics. The query must therefore cover
        # the larger of the two, and the results are filtered by type below.
        
        query_radius = max(self.point_sensing_radius, self.drone_sensing_radius)
        agents, distances = self.get_neighbors_in_radius(radius=query_radius)

        self.neighbors = []
        self.neighbor_distances = []
        self.perceived_points = []
        self.perceived_point_distances = []

        for agent, distance in zip(agents, distances):
            if isinstance(agent, BaseDrone) and distance <= self.drone_sensing_radius:
                self.neighbors.append(agent)
                self.neighbor_distances.append(float(distance))
            elif (isinstance(agent, TargetAgent) and distance <= self.point_sensing_radius):
                self.perceived_points.append(agent)
                self.perceived_point_distances.append(float(distance))

        # Perceived occupancy is built during the COMMUNICATE phase, using
        # only the drones with which communication is possible in this step.
        self.perceived_point_occupancies = []

    # ------------------------------------------------------------------
    # PHASE 2 - COMMUNICATION
    # ------------------------------------------------------------------

    def communicate(self):
        """Locally estimates how many drones are covering each perceived point.

        Communication deliberately remains high-level: no packets,
        protocols, or global point identifiers. For each point that I
        perceive, I count myself (if I cover it) and only the drones within ``drone_sensing_radius``
        whose communicated position falls within ``coverage_radius`` of that point.

        This way, two nearby points are not associated through a shared ID:
        the estimate is rebuilt at each step from the geometry of my local snapshot.
        """
        self.perceived_point_occupancies = []

        for point, my_distance in zip(self.perceived_points, self.perceived_point_distances):
            count = 1 if my_distance <= self.coverage_radius else 0

            for neighbor in self.neighbors:
                neighbor_point_distance = np.linalg.norm(neighbor.position - point.position) #this assumes that drones communicate their position accurately
                if neighbor_point_distance <= self.coverage_radius:
                    count += 1

            self.perceived_point_occupancies.append(count)

    # ------------------------------------------------------------------
    # PHASE 3 - TARGET SELECTION
    # ------------------------------------------------------------------

    def _estimated_occupancy(self, point):
        """Returns the estimate associated with the local detection of ``point``.

        The ``is`` comparison is not a communicated identifier: it is used only
        internally, within the same step, to retrieve from the local snapshot
        the TargetAgent object that the drone is already considering.
        """
        for candidate, occupancy in zip(self.perceived_points, self.perceived_point_occupancies):
            if candidate is point:
                return occupancy
        return 0

    def _perceived_deficit(self, point, distance):
        deficit = point.priority - self._estimated_occupancy(point)

        # If I am already covering the point, evaluate how many drones would be missing AFTER my possible departure.
        if distance <= self.coverage_radius:
            deficit += 1
        return deficit

    def _distance_to_perceived_point(self, point):
        for candidate, distance in zip(self.perceived_points, self.perceived_point_distances):
            if candidate is point:
                return distance
        return None

    def _current_target_is_overcrowded(self):
        """True only if I am covering my target and the estimate says occupancy > priority."""
        if self.target is None:
            return False

        distance = self._distance_to_perceived_point(self.target)
        if distance is None or distance > self.coverage_radius:
            return False

        return self._estimated_occupancy(self.target) > self.target.priority

    def _reset_release_wait(self):
        self.release_wait_remaining = None

    def _apply_random_release_wait(self, candidate_target):
        """Delays departure from an overcrowded target.

        This function is called only after establishing that:
        - a current target exists;
        - the new candidate differs from the current target;
        - the current target is perceived as overcrowded.
        """

        # Draw the delay only once.
        if self.release_wait_remaining is None:
            if self.release_delay_max_steps > 0:
                self.release_wait_remaining = int( self.model.rng.integers(1, self.release_delay_max_steps + 1) )
            else:
                self.release_wait_remaining = 0

        # Stay on the current target until the timer expires.
        if self.release_wait_remaining > 0:
            self.release_wait_remaining -= 1
            return self.target

        # Timer expired: the new decision can be applied.
        return candidate_target

    def decide_target(self):
        """Selects the target using only perceived/communicated information."""
        best_utility = -np.inf
        best_target = None

        for point, distance in zip(self.perceived_points, self.perceived_point_distances):
            deficit = self._perceived_deficit(point, distance)
            if deficit <= 0:
                continue

            utility = deficit - self.beta * distance
            if utility > best_utility:
                best_utility = utility
                best_target = point

        leaving_target = ( self.target is not None and best_target is not self.target )

        if leaving_target and self._current_target_is_overcrowded():
            best_target = self._apply_random_release_wait(best_target)
        else:
            self._reset_release_wait()

        self.planned_target = best_target
        self.planned_exploring = best_target is None

    # ------------------------------------------------------------------
    # PHASE 4 - STATIONING POLICY
    # ------------------------------------------------------------------

    def decide_station(self):
        """Shared hook: the fixed-wing drone has no discrete stationing policy."""
        pass

    # ------------------------------------------------------------------
    # PHASE 5 - DECISION COMMIT
    # ------------------------------------------------------------------

    def commit_decision(self):
        """Transforms the planned state into the current state."""
        previous_target = self.target
        self.target = self.planned_target
        self.exploring = self.planned_exploring

        if self.target is not previous_target:
            self._reset_release_wait()

    # ------------------------------------------------------------------
    # SHARED FORCES
    # ------------------------------------------------------------------

    def _separation_force(self):
        """Boids component that moves away from neighbors within ``separation``."""
        neighbor_count = len(self.neighbors)
        if neighbor_count == 0:
            return np.zeros(2)

        neighbor_deltas = self.space.calculate_difference_vector(
            self.position, agents=self.neighbors
        )

        separation_vector = np.zeros(2)
        separation_neighbor_count = 0


        for i in range(neighbor_count):
            if self.neighbor_distances[i] < self.separation:
                separation_vector -= neighbor_deltas[i]
                separation_neighbor_count += 1

        # No neighbor is close enough to activate separation.
        if separation_neighbor_count == 0:
            return np.zeros(2)

        # Keep the same contribution normalization used in the original
        # code: the sum is averaged over the total number of neighbors within drone_sensing_radius.
        return (separation_vector * self.separate_factor) / separation_neighbor_count

    def _alignment_force(self):
        """Boids component that tends to align the route with the neighbors' routes."""
        neighbor_count = len(self.neighbors)
        if neighbor_count == 0:
            return np.zeros(2)

        direction_sum = np.zeros(2)
        for neighbor in self.neighbors:
            direction_sum += neighbor.direction

        return (direction_sum * self.match_factor) / neighbor_count

    def _neighbor_force(self):
        """Convenience method for normal flight: separation + alignment."""
        return self._separation_force() + self._alignment_force()

    def _target_attraction(self):
        if self.target is None:
            return np.zeros(2)

        delta = self.target.position - self.position
        distance = np.linalg.norm(delta)
        if distance <= EPS:
            return np.zeros(2)
        return (delta / distance) * self.cohere_factor

    def _rotated_exploration_direction(self):
        angle = self.model.rng.normal(0, self.explore_factor)
        cos = np.cos(angle)
        sin = np.sin(angle)
        dx, dy = self.direction
        return np.array([(cos * dx) - (sin * dy), (sin * dx) + (cos * dy)])

    # ------------------------------------------------------------------
    # PHASE 6 - MOVEMENT: subclass contract
    # ------------------------------------------------------------------

    def move(self):
        raise NotImplementedError("Kinematics must be implemented by the subclass.")


class FixedWingDrone(BaseDrone):
    """Fixed-wing drone: constant speed and progressive steering."""

    drone_type = "fixed_wing"

    def move(self):
        
        steer = self._target_attraction()
        steer += self._separation_force()
        steer += self._alignment_force()
        steer += self._boundary_force()

        if self.exploring:
            rotated_direction = self._rotated_exploration_direction()
            steer += rotated_direction - self.direction

        # The new route is the previous route plus steer: this is where progressive
        # steering appears and, consequently, the turning-radius constraint.
        self.direction = self._normalize((self.direction + steer), fallback=self.direction)
        self._update_angle() # used for Solara visualization, does not affect movement
        self.position = self._clip_position(self.position + self.direction * self.speed)
        self.moving = True


class QuadcopterDrone(BaseDrone):
    """Quadcopter with FREE / OWNER / SUPPORT / DEPARTING roles.
        Overrides BaseDrone's communicate(), decide_target(), decide_station(), commit_decision(), and move() methods
    """

    drone_type = "quadcopter"

    def __init__(
        self,
        model,
        space,
        avoid_angle_degrees=10.0,
        support_inset=2.0,
        **kwargs,
    ):
        super().__init__(
            model=model,
            space=space,
            **kwargs,
        )

        # The various planned_* fields represent future decisions, 
        # and prevent drone execution order from affecting behavior.
        # The rule is: a drone must not read a field that others might still modify during the same phase.

        # Small deviation used when a nearby station reports that it is already satisfied.
        self.avoid_angle = np.deg2rad(float(avoid_angle_degrees))

        # Safety distance from the coverage boundary at which supports stop.
        # The operating radius is coverage_radius - support_inset.
        self.support_inset = float(support_inset)

        # Current and planned stationing role.
        # station_role = None -> free/traveling drone
        # station_role = "owner" -> owner stationary at the center
        # station_role = "support"  -> support stationary at the inner position
        self.station_role = None  
        self.planned_station_role = None

        # Value published exclusively by the owner during communicate().
        # Supports and explorers do not build their own deficit estimate.
        self.advertised_deficit = None

        # Transitional state: the drone is not yet a SUPPORT while reaching its inner radial position.
        # The decision to start relocation remains buffered through planned_support_relocation.
        self.support_destination = None # support_destination -> moving toward the support position
        self.planned_support_relocation = False

        # Radial direction of the side from which the support entered the coverage zone.
        self.entry_direction = None

        # Target that the drone is leaving.
        # None means that the drone is not DEPARTING.
        self.departing_from = None # departing_from -> moving radially away from a point
        self.planned_departing = False

        # Guidance received from a stationary drone reporting a positive deficit.
        # target = I directly see the point and can apply the stationing policy
        # guidance_position = I only know the direction toward a reported station and can only approach it
        # planned_target and planned_guidance_position -> never both present
        # target and guidance_position -> never both present after the commit
        # target and planned_guidance_position -> may both be temporarily present because they describe two different steps
        self.guidance_position = None # guidance_position -> follows the call of a station that is not directly visible
        self.planned_guidance_position = None

        # Position of a satisfied station from which to deviate slightly during exploration.
        self.avoid_position = None # avoid_position -> deviates slightly from a satisfied station
        self.planned_avoid_position = None

    @property
    def owner(self):
        return self.station_role == "owner"

    # ------------------------------------------------------------------
    # Quadcopter communication
    # ------------------------------------------------------------------

    def communicate(self):
        """Makes only the owner calculate and publish the deficit.

        FREE and SUPPORT do not estimate station occupancy. The owner, stationary at the
        center, counts itself and only the visible OWNER/SUPPORT drones that fall within
        its target's coverage. The result is published in
        ``advertised_deficit`` and will be read or relayed by other drones in the
        subsequent decision phases.
        """
        # The inherited field remains necessary for BaseDrone/FixedWingDrone, but is not
        # used by the quadcopter policy.
        self.perceived_point_occupancies = []
        self.advertised_deficit = None

        if self.station_role != "owner": # only the owner calculates the deficit
            return

        if self.departing_from is not None or self.target is None:
            return

        if not self._target_is_still_perceived(self.target):
            return

        my_distance = np.linalg.norm(self.position - self.target.position)
        if my_distance > self.coverage_radius:
            return

        # The owner counts itself.
        occupancy = 1

        for neighbor in self.neighbors:
            if not isinstance(neighbor, QuadcopterDrone):
                continue
            if neighbor.station_role not in ("owner", "support"):
                continue
            if neighbor.departing_from is not None:
                continue

            neighbor_point_distance = np.linalg.norm(neighbor.position - self.target.position)
            if neighbor_point_distance <= self.coverage_radius:
                occupancy += 1

        self.advertised_deficit = self.target.priority - occupancy

    # ------------------------------------------------------------------
    # Geometric point association
    # ------------------------------------------------------------------

    @staticmethod
    def _same_point(point_a, point_b):
        """Two points are considered equal through their position."""
        if point_a is None or point_b is None:
            return False

        return (np.linalg.norm(point_a.position - point_b.position) <= EPS)

    def _target_is_still_perceived(self, target):
        """Geometrically checks whether the target is still perceived."""
        if target is None:
            return False

        for point in self.perceived_points:
            if self._same_point(point, target):
                return True

        return False

    # ------------------------------------------------------------------
    # Currently covered points
    # ------------------------------------------------------------------

    def _covered_perceived_points(self):
        """Perceived points within coverage_radius."""
        covered = []

        for point, distance in zip(
            self.perceived_points,
            self.perceived_point_distances,
        ):
            if distance <= self.coverage_radius:
                covered.append((point, distance))

        return covered

    # ------------------------------------------------------------------
    # Station owner and deficit
    # ------------------------------------------------------------------

    def _owners_for_point(self, point):
        """Visible owners geometrically associated with ``point``.
        The point object or ID is neither communicated nor compared: every
        association arises from the geometric coincidence of positions.
        """
        owners = []

        if (self.station_role == "owner" and self.departing_from is None and self._same_point(self.target, point)):
            owners.append(self)

        for neighbor in self.neighbors:
            if not isinstance(neighbor, QuadcopterDrone):
                continue
            if neighbor.station_role != "owner":
                continue
            if neighbor.departing_from is not None:
                continue
            if self._same_point(neighbor.target, point):
                owners.append(neighbor)

        return owners

    def _find_owner_for_point(self, point):
        """Returns the authoritative station owner, if visible.

        If multiple owners exist, all drones that see them calculate the same
        winner: first the one closest to the center and, in case of a tie, the drone with
        the lower ``unique_id``.
        """
        owners = self._owners_for_point(point)

        if not owners:
            return None

        return min(
            owners,
            key = lambda drone: (np.linalg.norm(drone.position - point.position), drone.unique_id)
        )

    def _direct_point_information(self):
        """Evaluates perceived points first.

        For a staffed point, the owner's deficit is the only authority. A
        point without an owner remains eligible, and the election occurs only
        after entering its coverage.
        """
        information = []

        for point, distance in zip(self.perceived_points, self.perceived_point_distances):
            owner = self._find_owner_for_point(point)
            owner_deficit = None

            if owner is not None:
                owner_deficit = owner._station_deficit()

            information.append(
                {
                    "point": point,
                    "distance": distance,
                    "owner": owner,
                    "owner_deficit": owner_deficit,
                }
            )

        return information

    @staticmethod
    def _direct_point_rank(info):
        """Need, priority, and distance used to select a useful point."""
        point = info["point"]
        deficit = info["owner_deficit"]

        # No authoritative deficit exists for a point without an owner: 
        # priority represents the initial demand of the new station.
        need = point.priority if deficit is None else deficit

        return (need, point.priority, -info["distance"])

    def _station_deficit(self):
        """Returns the deficit published by the owner in communicate()."""
        if self.station_role != "owner":
            return None

        return self.advertised_deficit
    
    def _deficit_to_share(self):
        """Information that a stationary drone communicates to an explorer."""
        if self.station_role not in ("owner", "support"):
            return None

        if self.target is None:
            return None

        # Even an owner that is about to lose a conflict relays the deficit
        # of the authoritative owner, not its own competing estimate.
        owner = self._find_owner_for_point(self.target)

        if owner is None:
            return None

        return owner._station_deficit()

    # ------------------------------------------------------------------
    # Information received from stationary drones
    # ------------------------------------------------------------------

    def _stationary_information(self):
        # _stationary_information() builds this information: 
        #   - Which stations are communicating with me, and what is each point's authoritative message?
        # The result contains at most one message for each point,
        # even if the drone sees the owner and multiple supports of the same station.
        """Collects and filters overlapping messages from nearby stations.
            AT MOST one message for each point
        """

        # Valid messages are inserted here. Each element describes a station, not merely a drone.
        information = []

        for neighbor, distance in zip(self.neighbors, self.neighbor_distances):
            if not isinstance(neighbor, QuadcopterDrone):
                continue

            if neighbor.station_role not in ("owner", "support"):
                continue

            # Exclude a drone that is leaving the station.
            if neighbor.departing_from is not None:
                continue

            # Exclude a drone with no associated reference point
            if neighbor.target is None:
                continue

            deficit = neighbor._deficit_to_share()
            # NOTE: neighbor._deficit_to_share() behaves as follows: 
            # If the neighbor is the owner: it returns the deficit calculated and published by that owner
            # If the neighbor is a support: it geometrically identifies its point's owner and returns the deficit published by the owner
            # Therefore, the support does not communicate its own estimate. It only acts as a relay

            if deficit is None:
                continue

            message = {
                "drone": neighbor,
                "position": neighbor.target.position.copy(),
                "priority": neighbor.target.priority,
                "deficit": deficit,
                "distance": distance,
                "source_is_owner": neighbor.station_role == "owner",
            }

            existing_index = None
            for index, existing in enumerate(information):
                if (np.linalg.norm(existing["position"] - message["position"])<= EPS):
                    existing_index = index
                    break

            # If the point is not yet present, add the message because it is the first received for that station
            if existing_index is None:
                information.append(message)
                continue

            # If a message for that point already exists: 
            # decide whether to keep the stored message or replace it with the new one.
            existing = information[existing_index]
            prefer_new = (
                message["source_is_owner"] and not existing["source_is_owner"] # Choose the owner because it is the direct, authoritative source.
            ) or ( # If both sources are owners or both are supports, prefer the drone closest to us
                (message["source_is_owner"] == existing["source_is_owner"]) and (message["distance"] < existing["distance"])
            )

            if prefer_new:
                information[existing_index] = message

        return information

    # ------------------------------------------------------------------
    # Overcrowding timer
    # ------------------------------------------------------------------

    def _draw_overcrowding_wait(self):
        """Draws the waiting time of an overcrowded support."""
        if self.target is None:
            return 0

        distance = np.linalg.norm(self.position - self.target.position)

        exit_distance = max(0.0, self.coverage_radius - distance)

        minimum_wait = max(1,int(np.ceil(exit_distance/ max(self.speed, EPS))))

        # 1 near the center, 0 near the boundary.
        depth = np.clip(1.0 - (distance / self.coverage_radius), 0.0, 1.0)

        maximum_extra_wait = int(np.ceil(self.release_delay_max_steps* depth))

        # Even near the boundary, keep a small
        # pseudorandom interval if the parameter allows it.
        if self.release_delay_max_steps > 0:
            maximum_extra_wait = max(1, maximum_extra_wait)

        if maximum_extra_wait == 0:
            return minimum_wait

        return int(self.model.rng.integers(minimum_wait,minimum_wait + maximum_extra_wait + 1))

    def _support_should_depart(self, owner):
        """Manages the support's waiting period in case of overcrowding."""
        deficit = owner._station_deficit()

        # No reliable information, or the point is not overcrowded.
        if deficit is None or deficit >= 0:
            self._reset_release_wait()
            return False

        # First step in which overcrowding is detected: 
        # draw the timer only once and start waiting from the next step.
        if self.release_wait_remaining is None:
            self.release_wait_remaining = (self._draw_overcrowding_wait())
            return False

        if self.release_wait_remaining > 0:
            self.release_wait_remaining -= 1
            if self.release_wait_remaining > 0:
                return False

        # The timer has expired: explicitly reread the owner's current deficit
        # and depart only if overcrowding persists.
        final_deficit = owner._station_deficit()

        if final_deficit is not None and final_deficit < 0:
            return True

        self._reset_release_wait()
        return False

    # ------------------------------------------------------------------
    # Target decision
    # ------------------------------------------------------------------

    def decide_target(self):
        """Plans the quadcopter destination using local information.

        Preserves any ongoing transitions; otherwise, it first evaluates perceived
        points, using the owner's deficit as authoritative information, and then the
        messages from stationary drones. If it finds no useful requests, it plans
        exploration with a possible deviation from satisfied stations.
        The decision is stored in the planned_* fields and applied during the commit.
        """
        self.planned_guidance_position = None
        self.planned_avoid_position = None

        # It has already been established that this drone must become a support.
        # It has been assigned an inner coverage position but has not reached it yet.
        if self.support_destination is not None:
            self.planned_target = self.target
            self.planned_exploring = False
            return

        # Do not select new destinations during departure.
        if self.departing_from is not None:
            self.planned_target = None
            self.planned_exploring = False
            return

        # OWNER and SUPPORT retain their station while the point continues to exist.
        if self.station_role in ("owner", "support"):
            if self._target_is_still_perceived(self.target):
                self.planned_target = self.target
                self.planned_exploring = False
                return

            # The point is no longer perceived.
            self.planned_target = None
            self.planned_exploring = True
            self._reset_release_wait()
            return

        # 1) POINTS FIRST. 
        # For each staffed point, the deficit is communicated by the point's owner.
        point_information = self._direct_point_information()

        useful_points = []
        points_to_avoid = []

        for info in point_information:
            owner = info["owner"]
            owner_deficit = info["owner_deficit"]

            if owner is None:
                # A point without an owner can be selected. 
                # The owner will be elected only when the drone is actually within coverage_radius.
                useful_points.append(info)
            elif owner_deficit is not None and owner_deficit > 0:
                useful_points.append(info)
            else:
                # deficit <= 0 (or temporarily unavailable): do not
                # recalculate it locally; only avoid the station.
                points_to_avoid.append(info)

        if useful_points:
            choice = max(useful_points, key=lambda info: self._direct_point_rank(info))

            self.planned_target = choice["point"]
            self.planned_exploring = False
            self._reset_release_wait()
            return

        # 2) DRONES SECOND. 
        # Consider messages from nearby stationary drones only when there is no directly useful point.
        stationary_information = self._stationary_information()
        # Filter all point information with deficit > 0
        requests = [info for info in stationary_information if info["deficit"] > 0]

        if requests:
            choice = max(
                requests,
                key=lambda info: (info["deficit"], info["priority"], -info["distance"])
            )

            self.planned_target = None
            self.planned_exploring = False
            self.planned_guidance_position = choice["position"].copy()
            self._reset_release_wait()
            return

        
        # This block is reached only after the drone has verified that:
        #   1) no useful perceived points exist
        #   2) no calls from stationary drones with deficit > 0 exist
        # The drone must therefore explore, but it may know of satisfied points to avoid

        self.planned_target = None
        self.planned_exploring = True
        self._reset_release_wait()

        # Put points_to_avoid data in the same format as avoid_candidates
        avoid_candidates = [
            {
                "position": info["point"].position,
                "priority": info["point"].priority,
                "distance": info["distance"],
            }
            for info in points_to_avoid
        ]

        
        avoid_candidates.extend(
            {
                "position": info["position"],
                "priority": info["priority"],
                "distance": np.linalg.norm(self.position - info["position"])
            }
            for info in stationary_information if info["deficit"] <= 0
        )


        if avoid_candidates:
            # Select the nearest point as the position to avoid; at equal distance, avoid the higher-priority point
            avoid_choice = max(avoid_candidates, key=lambda info: (-info["distance"], info["priority"]))  
            self.planned_avoid_position = avoid_choice["position"].copy()

    # ------------------------------------------------------------------
    # OWNER / SUPPORT election
    # ------------------------------------------------------------------

    def decide_station(self):
        """
        Decides which stationing role the drone must have after commit_decision():
            -FREE
            -OWNER
            -SUPPORT
            -DEPARTING
            -relocation toward SUPPORT
        """
        # These are the values that decide_station will set
        self.planned_station_role = None # Role that the drone will have after the commit
        self.planned_departing = False  # Must it start leaving the station?
        self.planned_support_relocation = False # Must it reach the inner support position?

        # Already departing? -> do not decide roles
        if self.departing_from is not None:
            return

        # Moving toward the support position?
        # -> if arrived, become a support
        # -> otherwise, continue relocation
        if self.support_destination is not None:
            destination_distance = np.linalg.norm(self.position - self.support_destination)
            if destination_distance <= EPS:
                self.planned_station_role = "support"
            return

        # Am I the owner?
        # -> resolve any conflicts
        # -> if I win, remain owner
        # -> if I lose, relocate as support
        if self.station_role == "owner":
            if self._target_is_still_perceived(self.target):
                elected_owner = self._find_owner_for_point(self.target)

                if elected_owner is self: # VICTORY
                    # The owner never leaves due to overcrowding.
                    self.planned_station_role = "owner"
                elif elected_owner is not None: # DEFEAT
                    # A losing owner leaves the center and reaches the inner radial position before becoming SUPPORT.
                    self.planned_support_relocation = True

                self._reset_release_wait() # An owner must not retain any departure timer.
            return

        # Am I a support?
        # -> if the owner exists, check its deficit:
        #   -> nonnegative deficit: remain
        #   -> negative deficit: wait and possibly depart
        # -> if the owner does not exist, fall back to the election
        if self.station_role == "support":
            owner = self._find_owner_for_point(self.target)

            if owner is not None:
                self.planned_station_role = "support"

                if self._support_should_depart(owner):
                    self.planned_station_role = None # cancel the planned support role
                    self.planned_departing = True # plan the start of departure

                return
            # Owner not found: do not return.
            # The former support continues with the normal election.

        point = self.planned_target
        # From this point, the function handles free drones heading toward a point and former supports left without an owner.
        # It uses planned_target, not target, because it must work with the decision just made by decide_target().

        # Do I have a planned_target?
        # -> no: no election
        # -> yes: check whether I am within coverage
        if point is None:
            return

        my_distance = np.linalg.norm(self.position - point.position)

        # The drone can participate in the stationing policy only when physically within coverage.
        if my_distance > self.coverage_radius:
            return 
        
        # Does an owner already exist within coverage?
        # -> yes: relocate as support
        # -> no: compare all candidates
        existing_owner = self._find_owner_for_point(point)

        if existing_owner is not None:
            self.planned_support_relocation = True
            return

        # No owner: candidates continue toward the center. 
        # The election becomes effective only when the best candidate has reached the point exactly;
        # this prevents the owner from being frozen at the boundary.
        candidates = [self]

        for neighbor in self.neighbors:
            if not isinstance(neighbor, QuadcopterDrone):
                continue

            if neighbor.departing_from is not None: 
                continue

            if neighbor.planned_target is None:
                continue

            if not self._same_point(neighbor.planned_target, point):
                continue

            neighbor_distance = np.linalg.norm(neighbor.position - point.position)

            if neighbor_distance <= self.coverage_radius:
                candidates.append(neighbor)

        winner = min(candidates, key=lambda drone: (np.linalg.norm(drone.position - point.position), drone.unique_id))

        winner_distance = np.linalg.norm(winner.position - point.position)

        # Is the best candidate at the center?
        # -> no: continue approaching
        # -> yes:
        #   -> winner: owner
        #   -> others: support relocation
        if winner_distance > EPS:
            return

        if winner is self:
            self.planned_station_role = "owner"
        else:
            self.planned_support_relocation = True

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def commit_decision(self):

        """
        commit_decision() transforms planned_* decisions into the quadcopter's current state.
        It neither decides nor moves the drone; it applies what was established by:
        decide_target()
        decide_station()
        The function handles three main cases, in this order:
            1. start of departure
            2. start of support relocation
            3. normal commit
        """
        # A support has finished waiting (_support_should_depart() completed the wait) and must leave.
        if (self.planned_departing and self.departing_from is None):
            self.departing_from = self.target # store the point to leave

            # Clear the stationing state
            self.support_destination = None
            self.station_role = None 
            self.planned_station_role = None

            self.target = None
            self.planned_target = None
            self.guidance_position = None
            self.avoid_position = None

            # The drone is leaving the zone and is not yet considered to be exploring
            self.exploring = False

            self._reset_release_wait()
            return

        # The drone was already departing
        if self.departing_from is not None:
            return

        # This can happen when:
        #   -a drone enters coverage and finds an existing owner;
        #   -a candidate loses the election;
        #   -an owner loses a conflict among multiple owners.
        # The drone does not immediately become a support. It must first reach the inner radial position.
        if self.planned_support_relocation:

            if self.planned_target is None:
                raise RuntimeError("SUPPORT relocation planned without planned_target.")
            
            super().commit_decision()

            self.station_role = None
            self.planned_station_role = None
            self.guidance_position = None
            self.avoid_position = None
            self.exploring = False

            # Store the entry direction
            if self.entry_direction is None:
                self._remember_entry_direction(self.target)
                support_radius = max(0.0, self.coverage_radius - self.support_inset)
                destination = (self.target.position + (self.entry_direction * support_radius))
                self.support_destination = self._clip_position(destination)

            self._reset_release_wait()
            return


        # If the drone:
        #   -is not starting a departure;
        #   -is not already departing;
        #   -is not starting a relocation;
        #   -proceed with the normal commit.
        previous_role = self.station_role
        previous_target = self.target

        super().commit_decision()

        self.guidance_position = (None if self.planned_guidance_position is None else self.planned_guidance_position.copy())

        self.avoid_position = (None if self.planned_avoid_position is None else self.planned_avoid_position.copy())

        self.station_role = self.planned_station_role

        # new_station is true when, after the commit, the drone is owner/support and:
        # it was not stationary before, or it previously staffed a different point.
        new_station = (
            self.station_role in ("owner", "support")
            and (previous_role not in ("owner", "support") or not self._same_point(previous_target,self.target))
        )

        if new_station:
            # The radial direction is stored for any new role:
            # an owner can also become a support after an owner conflict.
            self._remember_entry_direction(self.target)

        # If the drone was previously owner/support and no longer is, clear the old entry direction.
        if self.station_role is None:
            if previous_role in ("owner", "support"):
                self.entry_direction = None

        # Clear movement state for stationary drones
        if self.station_role is not None:
            self.support_destination = None
            self.guidance_position = None
            self.avoid_position = None

    # ------------------------------------------------------------------
    # Entry-side memory
    # ------------------------------------------------------------------

    def _remember_entry_direction(self, point):
        delta = self.position - point.position
        magnitude = np.linalg.norm(delta)

        if magnitude > EPS:
            self.entry_direction = delta / magnitude
        else:
            self.entry_direction = self._normalize(-self.direction, fallback=np.array([1.0, 0.0]))

    # ------------------------------------------------------------------
    # Stationing
    # ------------------------------------------------------------------

    def _hold_station(self):
        """OWNER and SUPPORT remain exactly at the reached position."""
        self.moving = False

    def _move_exactly_towards_position(self, destination):
        """
            Moves toward a geometric position.
            If the distance to the point is shorter than the step, move only that distance to avoid overshooting
        """
        # Calculate vector and distance to the destination
        delta = destination - self.position
        distance = np.linalg.norm(delta)

        # Stop if the destination has already been reached
        if distance <= EPS:
            self.moving = False
            return True

        # Otherwise, orient the drone exactly toward the destination
        self.direction = delta / distance
        # Limit the step to the remaining distance
        step_length = min(self.speed, distance)

        self.position = self._clip_position(self.position + self.direction * step_length)
        self._update_angle()
        self.moving = step_length > EPS
        return distance <= self.speed + EPS

    def _finish_center_approach(self):
        """
        Snaps exactly to the center when the candidate is one step away.
        - if it returns False, no movement was performed;
        - if it performs the movement, it always returns True.
        """
        if self.target is None:
            return False

        delta = self.target.position - self.position
        distance = np.linalg.norm(delta)

        if distance > self.speed + EPS:
            return False

        return self._move_exactly_towards_position(self.target.position) 

    # ------------------------------------------------------------------
    # Departure from coverage
    # ------------------------------------------------------------------

    def _move_departure(self):
        # Retrieve the point being left
        point = self.departing_from

        if point is None:
            return

        # Recalculate entry_direction if it is missing
        if self.entry_direction is None:
            self._remember_entry_direction(point)

        # The drone adopts exactly the radial entry direction
        self.direction = self._normalize(self.entry_direction,fallback=self.direction)

        # Move one step along that direction.
        self.position = self._clip_position(self.position + self.direction * self.speed)

        # For graphical representation in the app
        self._update_angle()
        self.moving = True

        distance = np.linalg.norm(self.position - point.position)

        # Check whether the drone has left the stationing zone
        if distance > self.coverage_radius:
            self.departing_from = None
            self.entry_direction = None

            self.target = None
            self.guidance_position = None
            self.avoid_position = None

            self.exploring = True
            self._reset_release_wait()

    # ------------------------------------------------------------------
    # Guidance and deviation
    # ------------------------------------------------------------------

    def _attraction_to_position(self, position):
        delta = position - self.position
        distance = np.linalg.norm(delta)

        if distance <= EPS:
            return np.zeros(2)

        return (delta / distance) * self.cohere_factor

    def _avoidance_direction(self, position):
        """Slightly rotates the route away from a full station."""
        delta = position - self.position
        distance = np.linalg.norm(delta)

        if distance <= EPS:
            return self.direction.copy()

        toward_station = delta / distance

        # If already moving in the opposite direction, no further deviation is needed.
        if np.dot(self.direction, toward_station) <= 0:
            return self.direction.copy()

        cos = np.cos(self.avoid_angle)
        sin = np.sin(self.avoid_angle)

        dx, dy = self.direction

        left = np.array([(cos * dx) - (sin * dy), (sin * dx) + (cos * dy)])
        right = np.array([(cos * dx) + (sin * dy), (-sin * dx) + (cos * dy)])

        # Choose the rotation that points less directly toward the satisfied station.
        if (np.dot(left, toward_station) < np.dot(right, toward_station)):
            return left

        return right

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move(self):

        """
        move() executes the current state produced by commit_decision(),
        so it uses target, guidance_position, station_role, departing_from, and support_destination, 
        not their respective planned_* fields.
        Branch priority is:
            1. Departure from the station
            2. Support relocation
            3. Owner/support already stationary
            4. Final snap to the center
            5. Normal flight
        """
        # Transitional departure state.
        if self.departing_from is not None:
            self._move_departure()
            return

        # This state occurs when the drone:
        # has entered coverage;
        # has found an existing owner, or has lost the election;
        # must reach its inner support position.
        # Until it arrives, it is not yet a support: it is a relocating drone.
        if self.support_destination is not None:
            self._move_exactly_towards_position(self.support_destination)
            return

        # A stationing drone is stationary: 
        # no separation, return toward the center, or other movement within coverage.
        if self.station_role in ("owner", "support"):
            self._hold_station()
            return

        # A point without an owner requires the candidate to reach the center before the election. 
        # The final snap prevents overshoot/oscillations caused by the constant-length step.
        # False: the snap was not performed; continue with normal flight;
        # True: the drone was moved to the center; end move().
        if self.target is not None and self._finish_center_approach():
            return

        # Normal flight.
        neighbor_force = (self._separation_force() + self._alignment_force())

        boundary_force = self._boundary_force()

        if self.target is not None:
            desired_direction = (self._target_attraction() + neighbor_force + boundary_force)

        elif self.guidance_position is not None:
            desired_direction = (self._attraction_to_position(self.guidance_position) + neighbor_force + boundary_force)

        else:
            if self.avoid_position is not None:
                base_direction = (self._avoidance_direction(self.avoid_position))
            else:
                base_direction = (self._rotated_exploration_direction())

            desired_direction = (base_direction + neighbor_force + boundary_force)

        self.direction = self._normalize(desired_direction, fallback=self.direction)

        self._update_angle()

        self.position = self._clip_position(self.position + (self.direction * self.speed))

        self.moving = True
